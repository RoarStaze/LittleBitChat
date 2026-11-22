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
        print(json.dumps(run(args),indent=2))
        return 0
    except (TrainingError,ValueError,FileNotFoundError,ImportError,RuntimeError,OSError) as exc:
        print('LittleBit training error: '+str(exc),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
