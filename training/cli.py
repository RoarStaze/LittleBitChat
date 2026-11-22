"""Explicit offline LittleBit-2 layer compression CLI.

No automatic downloads, no made-up chat, no 27B end-to-end quality claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .layer_train import (PROJECTIONS, PREFIX, TrainingError, determine_rank,
                          load_projection, packed_cpu_matvec, save_export, train_one)


def parser():
    root=argparse.ArgumentParser(description='LittleBit-2 SVD + Joint-ITQ + layerwise QAT training (not full model training)')
    commands=root.add_subparsers(dest='mode',required=True)
    for mode in ('demo','layer'):
        p=commands.add_parser(mode,help='Synthetic proof of training' if mode=='demo' else 'Train one real local dense Gemma projection')
        p.add_argument('--out',type=Path,default=Path('training_runs'))
        p.add_argument('--steps',type=int,default=50)
        p.add_argument('--batch-size',type=int,default=32)
        p.add_argument('--rank',type=int,default=None)
        p.add_argument('--eff-bit',type=float,default=0.55)
        p.add_argument('--itq-iterations',type=int,default=20)
        p.add_argument('--lr',type=float,default=0.002)
        p.add_argument('--seed',type=int,default=42)
        p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        p.add_argument('--no-residual',action='store_true')
        p.add_argument('--resume',action='store_true')
        p.add_argument('--checkpoint-every',type=int,default=10)
        if mode=='layer':
            p.add_argument('--checkpoint',type=Path,required=True)
            p.add_argument('--layer',type=int,required=True)
            p.add_argument('--projection',choices=PROJECTIONS,required=True)
            p.add_argument('--max-teacher-mib',type=int,default=512)
    # Access to original Gemma model is gated by Google's license.
    # No pretrained weight download happens without --download.
    a = commands.add_parser("acquire",
        help="Plan or explicitly download ONE licensed original Google Gemma 3 weight shard")
    a.add_argument("--layer", type=int, default=0)
    a.add_argument("--projection", choices=PROJECTIONS, default="self_attn.q_proj")
    a.add_argument("--out", type=Path, default=Path("models/gemma-3-27b-it"))
    a.add_argument("--download", action="store_true",
                   help="Explicit permission to download ONE shard (several GiB)")
    a.add_argument("--max-shard-gib", type=float, default=8,
                   help="Maximum permitted shard size in GiB, default 8")

    diagnosis = commands.add_parser("diagnose",
        help="Read one real dense projection; compare randomized SVD and binary-init errors")
    diagnosis.add_argument("--checkpoint", type=Path, required=True)
    diagnosis.add_argument("--layer", type=int, default=0)
    diagnosis.add_argument("--projection", choices=PROJECTIONS, default="self_attn.q_proj")
    diagnosis.add_argument("--rank", type=int, default=128)
    diagnosis.add_argument("--seed", type=int, default=42)
    diagnosis.add_argument("--itq-iterations", type=int, default=20)
    diagnosis.add_argument("--samples", type=int, default=96,
                           help="Independent Gaussian probes; 96 matches the trainer audit count")
    diagnosis.add_argument("--max-teacher-mib", type=int, default=512)
    diagnosis.add_argument("--no-residual", action="store_true")

    preview = commands.add_parser("rank-plan",
        help="Preview physical BPW for different factor ranks without training or loading weights")
    preview.add_argument("--checkpoint", type=Path, required=True)
    preview.add_argument("--layer", type=int, default=0)
    preview.add_argument("--projection", choices=PROJECTIONS, default="self_attn.q_proj")
    preview.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128, 256])
    preview.add_argument("--target-bpw", type=float, default=0.55)
    preview.add_argument("--max-teacher-mib", type=int, default=512)
    preview.add_argument("--allow-test-shapes", action="store_true")
    preview.add_argument("--no-residual", action="store_true")

    # Multi-projection training is strictly an opt-in local teacher workflow.
    for mode in ("inventory", "batch"):
        q = commands.add_parser(mode, help=(
            "Inspect original Gemma projection headers without reading weights"
            if mode == "inventory" else
            "Resume-safe sequential QAT of selected dense Gemma projections"
        ))
        q.add_argument("--checkpoint", type=Path, required=True)
        q.add_argument("--layer-start", type=int, default=0)
        q.add_argument("--layer-end", type=int, default=62)
        q.add_argument("--projections", nargs="+", choices=PROJECTIONS, default=PROJECTIONS)
        q.add_argument("--max-teacher-mib", type=int, default=512)
        q.add_argument("--allow-test-shapes", action="store_true",
                       help="TEST FIXTURE ONLY: permit non-Gemma3-27B shapes; never establishes real-model quality")
        if mode == "batch":
            q.add_argument("--out", type=Path, default=Path("training_runs/batch"))
            q.add_argument("--rank", type=int, default=32)
            q.add_argument("--steps", type=int, default=50)
            q.add_argument("--batch-size", type=int, default=16)
            q.add_argument("--lr", type=float, default=0.002)
            q.add_argument("--itq-iterations", type=int, default=20)
            q.add_argument("--seed", type=int, default=42)
            q.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            q.add_argument("--no-residual", action="store_true")
            q.add_argument("--checkpoint-every", type=int, default=10)
            q.add_argument("--max-jobs", type=int, default=1,
                           help="Maximum selected projections processed this invocation (default 1)")
            q.add_argument("--max-holdout-nrmse", type=float, default=0.25,
                           help="Random-probe error threshold only; NOT language quality")
            q.add_argument("--resume", action="store_true")
    return root


def run(args):
    if args.mode=='demo':
        generator=torch.Generator(device='cpu').manual_seed(args.seed)
        teacher=torch.randn(24,32,generator=generator)*0.13
        layer=0;projection='self_attn.q_proj';synthetic=True
    else:
        teacher=load_projection(args.checkpoint,layer=args.layer,
                                projection=args.projection,max_teacher_mib=args.max_teacher_mib)
        layer=args.layer;projection=args.projection;synthetic=False
    rank=determine_rank(teacher.shape[1],teacher.shape[0],rank=args.rank,
                        eff_bit=args.eff_bit,residual=not args.no_residual)
    dest=args.out.resolve()
    if dest.exists() and not dest.is_dir():
        raise TrainingError('Output path exists and is not a directory')
    if dest.is_symlink():
        raise TrainingError('Refusing output symlink')
    dest.mkdir(parents=True,exist_ok=True)
    stem=f'layer_{layer:02d}_{projection.replace(".","_")}'
    state=dest/(stem+'.state.pt')
    output=dest/(stem+f'_step_{args.steps:06d}.safetensors')
    audit=dest/(stem+f'_step_{args.steps:06d}.json')
    if output.exists() or audit.exists():
        raise TrainingError('Target export already exists; change steps or output directory')
    result=train_one(teacher,rank=rank,steps=args.steps,batch_size=args.batch_size,
                     lr=args.lr,itq_iterations=args.itq_iterations,seed=args.seed,
                     residual=not args.no_residual,device=args.device,
                     save_state=state,resume=args.resume,checkpoint_every=args.checkpoint_every)
    model=result.pop('model')
    prefix=f'{PREFIX}.{layer}.{projection}'
    save_export(model,output,prefix)
    x=torch.linspace(-1,1,teacher.shape[1],dtype=torch.float32)
    with torch.no_grad():
        expected=model.to('cpu')(x[None])[0].numpy()
    observed=packed_cpu_matvec(output,prefix,x.numpy())
    max_abs=float(np.max(np.abs(expected-observed)))
    parity=bool(np.allclose(expected,observed,atol=1e-4,rtol=1e-4))
    if not parity:
        output.unlink(missing_ok=True)
        raise TrainingError('Packed export inference mismatch: refusing to keep checkpoint')
    report={
        'status':'LAYER_QAT_EXPORTED','backend':args.device,'layer':layer,'projection':projection,
        'teacher_is_synthetic':synthetic,'teacher_weights_loaded_from_real_checkpoint':not synthetic,
        'full_model_trained':False,'validated_language_quality':False,
        'packed_inference_verified':parity,'packed_inference_max_abs_error':max_abs,
        'packed_file':str(output),'audit_json':str(audit),**result,
    }
    audit.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.mode == "diagnose":
            from .diagnose import diagnose_projection
            report = diagnose_projection(
                args.checkpoint, layer=args.layer, projection=args.projection,
                rank=args.rank, seed=args.seed, itq_iterations=args.itq_iterations,
                residual=not args.no_residual, probe_count=args.samples,
                max_teacher_mib=args.max_teacher_mib)
            print(json.dumps(report, indent=2))
        elif args.mode == "rank-plan":
            from .rank_plan import build_rank_plan
            report = build_rank_plan(
                args.checkpoint, layer=args.layer, projection=args.projection,
                ranks=args.ranks, target_bpw=args.target_bpw,
                max_teacher_mib=args.max_teacher_mib,
                allow_test_shapes=args.allow_test_shapes,
                residual=not args.no_residual)
            print(json.dumps(report, indent=2))
        elif args.mode == "acquire":
            from .teacher_acquire import acquire_teacher_shard
            result = acquire_teacher_shard(
                out=args.out, layer=args.layer, projection=args.projection,
                download=args.download, max_shard_gib=args.max_shard_gib)
            print(json.dumps(result, indent=2))
        elif args.mode in ("inventory", "batch"):
            from .batch import scan_source, run_batch
            common = {
                "source": args.checkpoint,
                "layer_start": args.layer_start,
                "layer_end": args.layer_end,
                "projections": args.projections,
                "max_teacher_mib": args.max_teacher_mib,
                "allow_test_shapes": args.allow_test_shapes,
            }
            if args.mode == "inventory":
                result = scan_source(**common)
            else:
                result = run_batch(**common, out=args.out, rank=args.rank,
                                   steps=args.steps, batch_size=args.batch_size,
                                   lr=args.lr, itq_iterations=args.itq_iterations,
                                   seed=args.seed, device=args.device,
                                   residual=not args.no_residual,
                                   checkpoint_every=args.checkpoint_every,
                                   max_jobs=args.max_jobs,
                                   max_holdout_nrmse=args.max_holdout_nrmse,
                                   resume=args.resume)
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(run(args),indent=2))
        return 0
    except (TrainingError,ValueError,FileNotFoundError,ImportError,RuntimeError,OSError) as exc:
        print('LittleBit training error: '+str(exc),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
