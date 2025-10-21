import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

import boltz.model.layers.initialize as init


@torch.compiler.disable
def kernel_attention_pair_bias(
    s,
    q,
    k,
    v,
    z,
    mask,
    num_heads,
    w_proj_z,
    w_proj_g,
    w_proj_o,
    w_ln_z=None,
    b_ln_z=None,
    b_proj_g=None,
    b_proj_o=None,
    inf=1e6,
    eps=1e-5,
):
    """Wrapper for NVIDIA cuEquivariance attention_pair_bias kernel.

    Uses optimized Triton kernels for long sequences and PyTorch fallback for short sequences.
    """
    from cuequivariance_torch import attention_pair_bias

    # Call NVIDIA's optimized kernel
    # Note: cuEquivariance kernel returns (output, proj_z)
    output, _ = attention_pair_bias(
        s=s,
        q=q,
        k=k,
        v=v,
        z=z,
        mask=mask,
        num_heads=num_heads,
        w_proj_z=w_proj_z,
        w_proj_g=w_proj_g,
        w_proj_o=w_proj_o,
        w_ln_z=w_ln_z,
        b_ln_z=b_ln_z,
        b_proj_z=None,  # Boltz doesn't use bias for z projection
        b_proj_g=b_proj_g,
        b_proj_o=b_proj_o,
        inf=inf,
        eps=eps,
        attn_scale=None,  # Uses default 1/sqrt(head_dim)
        return_z_proj=False,  # We don't need the projected z for now
        is_cached_z_proj=False,  # z is not pre-projected
    )

    return output


class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int
            The input pairwise dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6
        initial_norm: bool, optional
            Whether to apply layer norm to the input, by default True

        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.initial_norm = initial_norm
        if self.initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.proj_z = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, num_heads, bias=False),
            Rearrange("b ... h -> b h ..."),
        )

        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        init.final_init_(self.proj_o.weight)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        multiplicity: int = 1,
        to_keys=None,
        model_cache=None,
        use_kernels: bool = False,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The input sequence tensor (B, S, D)
        z : torch.Tensor
            The input pairwise tensor (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N)
        multiplicity : int, optional
            The diffusion batch size, by default 1
        use_kernels : bool, optional
            Whether to use NVIDIA cuEquivariance optimized kernels, by default False

        Returns
        -------
        torch.Tensor
            The output sequence tensor.

        """
        B = s.shape[0]

        # Layer norms
        if self.initial_norm:
            s = self.norm_s(s)

        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        # Handle z projection and caching
        z_is_cached = model_cache is not None and "z" in model_cache

        if use_kernels:
            print("Using NVIDIA cuEquivariance optimized kernel for AttentionPairBias")
            # NVIDIA cuEquivariance kernel path
            # The kernel expects z in (B, U, V, z_dim) format and handles projection internally

            # Get z projection layer components
            z_ln = self.proj_z[0]  # LayerNorm
            z_linear = self.proj_z[1]  # Linear projection

            # Prepare z: if cached, it's already projected; otherwise use raw z
            if z_is_cached:
                z_input = model_cache["z"]
                # Already projected: (B, H, U, V)
                is_cached_proj = True
            else:
                # Not projected yet: (B, U, V, z_dim)
                z_input = z
                is_cached_proj = False

            # Repeat for multiplicity (diffusion steps)
            z_input = z_input.repeat_interleave(multiplicity, 0)

            # Transpose q, k, v to match kernel expectations: (B*M, H, S, DH)
            q_kernel = q.transpose(1, 2).contiguous()
            k_kernel = k.transpose(1, 2).contiguous()
            v_kernel = v.transpose(1, 2).contiguous()

            # Call NVIDIA optimized kernel
            o = kernel_attention_pair_bias(
                s=s,
                q=q_kernel,
                k=k_kernel,
                v=v_kernel,
                z=z_input,
                mask=mask,
                num_heads=self.num_heads,
                w_proj_z=z_linear.weight if not is_cached_proj else None,
                w_proj_g=self.proj_g.weight,
                w_proj_o=self.proj_o.weight,
                w_ln_z=z_ln.weight if not is_cached_proj else None,
                b_ln_z=z_ln.bias if not is_cached_proj else None,
                b_proj_g=None,  # Boltz doesn't use bias for gating
                b_proj_o=None,  # Boltz doesn't use bias for output
                inf=self.inf,
                eps=1e-5,
            )

            # Cache z projection if needed
            if not z_is_cached and model_cache is not None:
                # Note: kernel returns projected z, but we don't use return_z_proj for now
                # In future, we could cache the projected z from the kernel
                model_cache["z"] = self.proj_z(z)

            return o

        # Original PyTorch path (fallback)
        # Caching z projection during diffusion roll-out
        if not z_is_cached:
            z = self.proj_z(z)

            if model_cache is not None:
                model_cache["z"] = z
        else:
            z = model_cache["z"]
        z = z.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        with torch.autocast("cuda", enabled=False):
            # Compute attention weights
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            attn = attn / (self.head_dim**0.5) + z.float()
            # The pairwise mask tensor (B, N) is broadcasted to (B, 1, 1, N) and (B, H, N, N)
            attn = attn + (1 - mask[:, None, None].float()) * -self.inf
            attn = attn.softmax(dim=-1)

            # Compute output
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)

        return o
