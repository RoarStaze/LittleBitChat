"""Layerwise LittleBit-2 research training; one real dense teacher matrix at a time.

Independent implementation of Samsung's publicly documented SVD + Joint-ITQ,
Dual-SVID-like magnitude scales, SmoothSign QAT and optional residual path.
This is not a claim of equivalent paper-level results or full-model training.
Models learned with random input probes are layer approximations, not a
quality-preserving LLM. No checkpoints or networks are accessed implicitly.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import os

import numpy as np
import torch
from torch import nn

PROJECTIONS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
               'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
PREFIX = 'language_model.model.layers'


class TrainingError(ValueError):
    pass


def determine_rank(input_features, output_features, *, eff_bit=0.55, residual=True,
                   min_rank=8, rank=None):
    for val in (input_features, output_features, min_rank):
        if type(val) is not int or val < 1:
            raise ValueError('Dimensions and minimum rank must be positive integers')
    if rank is not None:
        if type(rank) is not int or not 1 <= rank <= min(input_features, output_features):
            raise ValueError('Explicit rank exceeds input/output dimensions')
        return rank
    if type(eff_bit) not in (float, int) or not math.isfinite(eff_bit) or eff_bit <= 0 or eff_bit > 16:
        raise ValueError('eff_bit must be a finite positive target in (0,16]')
    a, b = input_features, output_features
    base = a + b + 16
    if residual:
        estimate = (a*b*eff_bit - 32*(a+b)) / (2 * base)
    else:
        estimate = (a*b*eff_bit - 16*(a+b)) / base
    rounded = max(min_rank, int(estimate // 8) * 8)
    if rounded > min(a, b):
        rounded = min(a, b)
    return rounded


def estimated_effective_bits(out_features, in_features, rank, residual):
    a, b, r = in_features, out_features, rank
    return (r*(a+b+16)*(2 if residual else 1) + (32 if residual else 16)*(a+b))/(a*b)


def _rank_one_scales(magnitude, iterations=15):
    v = torch.full((magnitude.shape[1],), 1/math.sqrt(magnitude.shape[1]), device=magnitude.device)
    for _ in range(iterations):
        u = magnitude @ v
        u = u / u.norm().clamp(min=1e-12)
        v = magnitude.t() @ u
        v = v / v.norm().clamp(min=1e-12)
    u = magnitude @ v
    sigma = u.norm().clamp(min=1e-12)
    u = u / sigma
    s = torch.sqrt(sigma)
    return u*s, v*s


@torch.no_grad()
def initialize(weight: torch.Tensor, *, rank, itq_iterations=20, seed=42):
    """SVD factors, then joint orthogonal ITQ; U@V is rotation-invariant."""
    if weight.ndim != 2 or not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise TrainingError('Finite 2D floating-point teacher matrix required')
    if type(rank) is not int or not 1 <= rank <= min(weight.shape):
        raise TrainingError('Rank outside teacher dimensions')
    if type(itq_iterations) is not int or not 0 <= itq_iterations <= 100:
        raise TrainingError('ITQ iterations must be 0..100')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        a, s, v = torch.svd_lowrank(weight.float(), q=rank, niter=3)
    u = a[:, :rank] * torch.sqrt(s[:rank])[None, :]
    vt = torch.sqrt(s[:rank])[:, None] * v[:, :rank].t()
    if itq_iterations:
        joint = torch.cat([u, vt.t()], dim=0)
        rand = torch.randn(rank, rank, generator=generator, dtype=torch.float32)
        rotation = torch.linalg.qr(rand)[0].to(weight.device)
        for _ in range(itq_iterations):
            binary = torch.where(joint @ rotation >= 0, 1., -1.)
            mat = binary.t() @ joint
            uu, _, vv = torch.linalg.svd(mat, full_matrices=False)
            rotation = vv.t() @ uu.t()
        u = u @ rotation
        vt = rotation.t() @ vt
    return u, vt


class SmoothSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1., -1.)

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        alpha = 100.
        return gradient * alpha * (1. - torch.tanh(alpha*x).square())


class LittleBitQAT(nn.Module):
    def __init__(self, components, *, bias=None):
        super().__init__()
        for suffix in ('U', 'V', 'u1', 'u2', 'v1', 'v2'):
            self.register_parameter(suffix, nn.Parameter(components[suffix].clone().float()))
        self.residual = 'U_R' in components
        if self.residual:
            for suffix in ('U_R','V_R','u1_R','u2_R','v1_R','v2_R'):
                self.register_parameter(suffix, nn.Parameter(components[suffix].clone().float()))
        self.register_buffer('bias', None if bias is None else bias.detach().clone().float())

    @staticmethod
    def _initialize_path(weight, rank, itq_iterations, seed):
        u,v = initialize(weight, rank=rank, itq_iterations=itq_iterations, seed=seed)
        u1,u2 = _rank_one_scales(u.abs())
        v1,v2 = _rank_one_scales(v.abs())
        return {'U':u, 'V':v, 'u1':u1, 'u2':u2, 'v1':v1, 'v2':v2}

    @classmethod
    def from_teacher(cls, weight, *, rank, itq_iterations=20, residual=True, seed=42, bias=None):
        components = cls._initialize_path(weight.float(), rank, itq_iterations, seed)
        if residual:
            with torch.no_grad():
                approx = cls(components).dense_weight()
                delta = weight.float() - approx
                second = cls._initialize_path(delta, rank, itq_iterations, seed+1)
                components.update({name+'_R':data for name,data in second.items()})
        return cls(components, bias=bias)

    def _forward_path(self, x, suffix=''):
        U,V = getattr(self,'U'+suffix),getattr(self,'V'+suffix)
        u1,u2,v1,v2 = [getattr(self,name+suffix) for name in ('u1','u2','v1','v2')]
        h = (x*v2) @ SmoothSign.apply(V).t()
        return ((h*(v1*u2)) @ SmoothSign.apply(U).t())*u1

    def forward(self, x):
        y = self._forward_path(x)
        if self.residual:
            y = y + self._forward_path(x, '_R')
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def dense_path_weight(self, suffix=""):
        """Materialize ONE signed path for bounded teacher calibration only."""
        if suffix not in ("", "_R") or (suffix == "_R" and not self.residual):
            raise TrainingError("Invalid signed path suffix")
        U, V = getattr(self, 'U'+suffix), getattr(self, 'V'+suffix)
        u1, u2, v1, v2 = [
            getattr(self, name+suffix) for name in ('u1','u2','v1','v2')
        ]
        return ((SmoothSign.apply(U) * (u1[:,None]*u2[None,:])) @
                (SmoothSign.apply(V) * (v1[:,None]*v2[None,:])))

    @torch.no_grad()
    def dense_weight(self):
        """Only for training one bounded teacher matrix, never inference."""
        result = self.dense_path_weight("")
        return result + self.dense_path_weight("_R") if self.residual else result


def _teacher_sha(weight):
    w=weight.detach().to('cpu', dtype=torch.float32).contiguous()
    return hashlib.sha256(w.numpy().tobytes()).hexdigest()


def _save_state(path, payload):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.is_symlink():
        raise TrainingError('Refusing to overwrite symlink training state')
    tmp=path.with_name(path.name+'.tmp')
    torch.save(payload,tmp)
    os.replace(tmp,path)


def train_one(weight, *, rank, steps=100, batch_size=32, lr=0.002,
              itq_iterations=20, residual=True, seed=42, device='cpu',
              save_state=None, resume=False, checkpoint_every=25,
              refine_output_scales=False, binary_refine_channels=0):
    """QAT ONE projection against a frozen dense teacher on random probes.

    NOT full-LLM QAT and cannot independently create a conversational model.
    """
    if type(steps) is not int or not 1<=steps<=100000:
        raise TrainingError('steps outside 1..100000')
    if type(batch_size) is not int or not 1<=batch_size<=4096:
        raise TrainingError('batch_size outside 1..4096')
    if type(checkpoint_every) is not int or not 1<=checkpoint_every<=10000:
        raise TrainingError('checkpoint_every outside 1..10000')
    if type(refine_output_scales) is not bool:
        raise TrainingError('refine_output_scales must be boolean')
    if type(binary_refine_channels) is not int or not 0 <= binary_refine_channels <= 256:
        raise TrainingError('binary_refine_channels must be an integer in 0..256')
    if type(lr) not in (int,float) or not math.isfinite(lr) or not 0<lr<=1:
        raise TrainingError('Invalid learning rate')
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise TrainingError('Requested PyTorch device is unavailable')
    teacher=weight.detach().float().to(device)
    digest=_teacher_sha(weight)
    checkpoint=Path(save_state) if save_state else None
    resumed=0
    if resume:
        if checkpoint is None or not checkpoint.is_file():
            raise TrainingError('No existing state checkpoint to resume')
        data=torch.load(checkpoint,map_location=device,weights_only=True)
        if (data.get('teacher_sha256')!=digest or data.get('rank')!=rank or
            data.get('seed')!=seed or data.get('residual')!=residual or
            data.get('itq_iterations')!=itq_iterations or data.get('batch_size')!=batch_size or
            data.get('lr')!=lr or
            data.get('refine_output_scales', False)!=refine_output_scales or
            data.get('binary_refine_channels', 0)!=binary_refine_channels):
            raise TrainingError('Resume state training/teacher mismatch')
        resumed=data['step']
        if type(resumed) is not int or steps<=resumed:
            raise TrainingError('Requested steps must exceed checkpoint step')
        # Reuse already-trained factors and optimizer; no expensive SVD rerun.
        # Only load saved states created by this project (weights_only=True).
        model=LittleBitQAT(data['model']).to(device)
        model.load_state_dict(data['model'])
    else:
        model=LittleBitQAT.from_teacher(teacher,rank=rank,itq_iterations=itq_iterations,
                                        residual=residual,seed=seed).to(device)
        if refine_output_scales:
            from .scale_refinement import refine_output_row_scales
            refine_output_row_scales(model, teacher)
        if binary_refine_channels:
            from .binary_refinement import refine_binary_signs
            refine_binary_signs(model, teacher, channels=binary_refine_channels)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    if resume:
        opt.load_state_dict(data['optimizer'])
    # The selection probes and final audit probes are deliberately independent.
    # Selecting the lowest error on the same holdout reported to users would
    # give optimistically biased quality figures.
    gen=torch.Generator(device='cpu').manual_seed(seed+10001)
    val_gen=torch.Generator(device='cpu').manual_seed(seed+8891)
    audit_gen=torch.Generator(device='cpu').manual_seed(seed+8892)
    validation=torch.randn(96,weight.shape[1],generator=val_gen).to(device)
    audit=torch.randn(96,weight.shape[1],generator=audit_gen).to(device)
    with torch.no_grad():
        validation_target=validation@teacher.t()
        audit_target=audit@teacher.t()
        current_validation_rmse=float(torch.sqrt(
            torch.mean((model(validation)-validation_target).square())).item())
        current_audit_rmse=float(torch.sqrt(
            torch.mean((model(audit)-audit_target).square())).item())
    if resume:
        # Old saved optimizers did not track the best weights or independent
        # baseline; treat their current state as a conservative baseline.
        original_audit_rmse=float(data.get('initial_holdout_rmse', current_audit_rmse))
        initial_validation_rmse=float(data.get('initial_validation_rmse', current_validation_rmse))
        best_step=int(data.get('best_step', resumed))
        best_validation_rmse=float(data.get('best_validation_rmse', current_validation_rmse))
        best_snapshot=data.get('best_model', None)
        if best_snapshot is None:
            best_snapshot={name:t.detach().cpu().clone() for name,t in model.state_dict().items()}
        if not math.isfinite(best_validation_rmse):
            raise TrainingError('Nonfinite best checkpoint validation score')
    else:
        original_audit_rmse=current_audit_rmse
        initial_validation_rmse=current_validation_rmse
        best_validation_rmse=current_validation_rmse
        best_step=0
        best_snapshot={name:t.detach().cpu().clone() for name,t in model.state_dict().items()}
    last_validation_rmse=current_validation_rmse
    for step in range(resumed,steps):
        if step==resumed and resumed:
            for _ in range(resumed):
                torch.randn(batch_size,weight.shape[1],generator=gen)
        x=torch.randn(batch_size,weight.shape[1],generator=gen).to(device)
        with torch.no_grad(): target=x@teacher.t()
        prediction=model(x)
        loss=(prediction-target).square().mean()
        if not torch.isfinite(loss):
            raise TrainingError('Nonfinite QAT loss; refusing to export corrupted factors')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        opt.step()
        at=step+1
        if at % checkpoint_every == 0 or at == steps:
            with torch.no_grad():
                last_validation_rmse=float(torch.sqrt(torch.mean(
                    (model(validation)-validation_target).square())).item())
            if not math.isfinite(last_validation_rmse):
                raise TrainingError('Nonfinite held-out validation score')
            if last_validation_rmse < best_validation_rmse:
                best_validation_rmse=last_validation_rmse
                best_step=at
                best_snapshot={name:t.detach().cpu().clone()
                               for name,t in model.state_dict().items()}
            if checkpoint is not None:
                _save_state(checkpoint, {'version':2,'step':at,'teacher_sha256':digest,
                    'rank':rank,'seed':seed,'residual':residual,
                    'itq_iterations':itq_iterations,'batch_size':batch_size,'lr':lr,
                    'refine_output_scales': refine_output_scales,
                    'binary_refine_channels': binary_refine_channels,
                    'model':model.state_dict(),'optimizer':opt.state_dict(),
                    'best_model':best_snapshot,'best_step':best_step,
                    'best_validation_rmse':best_validation_rmse,
                    'initial_validation_rmse':initial_validation_rmse,
                    'initial_holdout_rmse':original_audit_rmse})
    # The optimizer is checkpointed at the LAST step for resume. Only the
    # best held-out validation factors are selected for export.
    model.load_state_dict(best_snapshot)
    with torch.no_grad():
        rmse=float(torch.sqrt(torch.mean((model(audit)-audit_target).square())).item())
        denom=float(torch.sqrt(torch.mean(audit_target.square())).item())
    return {'model':model,'step':steps,'resumed_from_step':resumed,
            'best_step':best_step,'best_validation_rmse':best_validation_rmse,
            'initial_validation_rmse':initial_validation_rmse,
            'last_step_validation_rmse':last_validation_rmse,
            'independent_audit_probes':True,
            'initial_holdout_rmse':original_audit_rmse,'holdout_rmse':rmse,
            'holdout_nrmse':rmse/max(denom,1e-12),
            'holdout_error_regressed':rmse > original_audit_rmse,
            'teacher_sha256':digest,
            'refine_output_scales': refine_output_scales,
            'binary_refine_channels': binary_refine_channels,
            'rank':rank,'actual_effective_bits':estimated_effective_bits(*teacher.shape,rank,residual),
            'checkpoint':str(checkpoint) if checkpoint else None,
            'method':'layerwise_random_probe_qat_not_full_model_training'}


def _packed_int32(signs:np.ndarray):
    if signs.ndim!=2:
        raise TrainingError('Expected 2D factor matrix')
    negatives=(signs<0).astype(np.uint8)
    words=np.zeros((signs.shape[0],(signs.shape[1]+31)//32),dtype=np.uint32)
    for bit in range(signs.shape[1]):
        words[:,bit//32]|=negatives[:,bit].astype(np.uint32)<<np.uint32(bit%32)
    return words.view(np.int32)


def save_export(model:LittleBitQAT,path,prefix):
    """Produce Samsung-compatible real packed factor fields, not fake text."""
    from safetensors.numpy import save_file
    if not isinstance(model,LittleBitQAT):raise TypeError('Expected LittleBitQAT')
    path=Path(path)
    if path.exists():raise FileExistsError('Refusing to overwrite already exported factors')
    tensors={}
    for suffix in ('','_R') if model.residual else ('',):
        for name in ('U','V'):
            data=getattr(model,name+suffix).detach().cpu().float().numpy()
            key=name+suffix
            tensors[f'{prefix}.{key}_packed']=_packed_int32(data)
            tensors[f'{prefix}.{key}_shape']=np.asarray(data.shape,dtype='<i8')
        for name in ('u1','u2','v1','v2'):
            data=getattr(model,name+suffix).detach().cpu().float().numpy()
            tensors[f'{prefix}.{name+suffix}']=np.ascontiguousarray(data,dtype='<f4')
    if model.bias is not None:
        tensors[f'{prefix}.bias']=model.bias.detach().cpu().float().numpy().copy()
    path.parent.mkdir(parents=True,exist_ok=True)
    save_file(tensors,str(path), metadata={'training_method':'single_layer_qat','complete_model':'false'})
    return path


def packed_cpu_matvec(path,prefix,vector):
    """Independent exported signed packed inference arithmetic."""
    from safetensors import safe_open
    with safe_open(str(path),framework='np',device='cpu') as f:
        keys=set(f.keys())
        x=np.asarray(vector,dtype=np.float32)
        def one(suffix):
            def unpack(name):
                words=f.get_tensor(prefix+'.'+name+suffix+'_packed').view(np.uint32)
                shape=f.get_tensor(prefix+'.'+name+suffix+'_shape')
                cols=int(shape[1]);bits=np.arange(cols,dtype=np.uint32)
                return np.where(((words[:,bits//32]>>(bits%32))&1)!=0,-1.,1.).astype(np.float32)
            U,V=unpack('U'),unpack('V')
            u1,u2,v1,v2=[f.get_tensor(prefix+'.'+n+suffix).astype(np.float32).reshape(-1) for n in ('u1','u2','v1','v2')]
            return (((x*v2)@V.T)*(v1*u2))@U.T*u1
        y=one('')
        if prefix+'.U_R_packed' in keys:y=y+one('_R')
        if prefix+'.bias' in keys:y=y+f.get_tensor(prefix+'.bias')
        return y


def load_projection(source, *, layer, projection, max_teacher_mib=512):
    """Load ONE real HF Gemma3 dense matrix from local Safetensors, no network."""
    if type(layer) is not int or not 0<=layer<62:
        raise ValueError('layer must be in 0..61')
    if projection not in PROJECTIONS:
        raise ValueError('Invalid Gemma projection module')
    if type(max_teacher_mib) is not int or not 1<=max_teacher_mib<=4096:
        raise ValueError('Invalid source tensor byte budget')
    source = Path(source)
    if not source.exists():
        raise TrainingError('Local dense teacher checkpoint not found: '+str(source))
    from safetensors import safe_open
    key=f'{PREFIX}.{layer}.{projection}.weight'
    if source.is_file():
        shard=source
        if source.suffix!='.safetensors':
            raise TrainingError('Expected .safetensors file')
    elif source.is_dir():
        index=source/'model.safetensors.index.json'
        if not index.is_file() or index.stat().st_size>16*1024*1024:
            raise TrainingError('Missing or excessive sharded checkpoint manifest')
        import json
        try:
            manifest=json.loads(index.read_text(encoding='utf8'))
            filename=manifest['weight_map'][key]
        except (ValueError,KeyError,TypeError) as exc:
            raise TrainingError('Dense teacher key missing from checkpoint index') from exc
        if (not isinstance(filename,str) or not filename.endswith('.safetensors') or
                filename!=Path(filename).name or '/' in filename or chr(92) in filename):
            raise TrainingError('Unsafe shard filename in teacher manifest')
        shard=source/filename
        if not shard.is_file() or not shard.resolve().is_relative_to(source.resolve()):
            raise TrainingError('Dense teacher shard missing or unsafe path')
    else:
        raise TrainingError('Unsupported dense teacher path')
    try:
        with safe_open(str(shard),framework='pt',device='cpu') as file:
            if key not in file.keys():
                raise TrainingError('Dense teacher projection tensor missing from shard')
            desc=file.get_slice(key)
            shape=desc.get_shape()
            dtype=desc.get_dtype()
            bytes_per={'F16':2,'BF16':2,'F32':4}.get(dtype)
            if bytes_per is None or len(shape)!=2 or not all(type(n) is int and n>0 for n in shape):
                raise TrainingError('Expected finite floating 2D teacher projection')
            if shape[0]*shape[1]*bytes_per>max_teacher_mib*1024*1024:
                raise TrainingError('Dense teacher projection exceeds selected tensor budget')
            tensor=file.get_tensor(key).float()
        if not torch.isfinite(tensor).all():
            raise TrainingError('Dense teacher contains nonfinite values')
        return tensor
    except (OSError, ValueError) as exc:
        if isinstance(exc,TrainingError):
            raise
        raise TrainingError('Unable to read dense teacher projection: '+str(exc)) from exc
