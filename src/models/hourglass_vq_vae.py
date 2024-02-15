#@markdown Model.

from typing import List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce


class LinearUpsample(nn.Module):
    """Linear upsample.

    Example
    -------
    >>> module = LinearUpsample(embedding_dimension=256)
    >>> x = torch.randn((1, 5, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.linear = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension * 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.linear(x)
        x = rearrange(x, '... t (n e) -> ... (n t) e', n=2)

        return x


class LinearDownsample(nn.Module):
    """Linear downsample.

    Example
    -------
    >>> module = LinearDownsample(embedding_dimension=256)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 5, 256).
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.linear = nn.Linear(
            in_features=embedding_dimension * 2,
            out_features=embedding_dimension,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = rearrange(x, '... (n t) e -> ... t (n e)', n=2)
        x = self.linear(x)

        return x


class RoPE(nn.Module):
    """Rotary positional embedding.
    
    Example
    -------
    >>> module = RoPE(embedding_dimension=256, base=10_000)
    >>> q = torch.randn((1, 10, 256))
    >>> k = torch.randn((1, 10, 256))
    >>> alignment = torch.einsum('bte,bse->bts', module(q), module(k))
    """

    def __init__(self, *, embedding_dimension: int, base: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        base : int
            The base to use for absolute positional encodings.
        """

        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.base = base

        # Precompute theta.

        exponent = torch.arange(
            start=0,
            end=embedding_dimension,
            step=2,
            dtype=torch.float,
        ) / embedding_dimension

        theta = 1. / torch.pow(base, exponent)

        self.theta = theta

    def absolute_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Perform absolute positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        encoding : torch.Tensor
            The absolute positional encoding.
        """

        if self.theta.device != x.device:
            self.theta = self.theta.to(x.device)

        encoding = torch.einsum(
            't,e->te',
            torch.arange(x.size(-2), dtype=torch.float, device=x.device),
            self.theta,
        )

        encoding = repeat(encoding, '... e -> ... (e n)', n=2)

        return encoding

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate each subspace by -90 degrees."""

        x = rearrange(x, '... (e n) -> ... e n', n=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = rearrange(x, '... e n -> ... (e n)')

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Foward pass."""

        encoding = self.absolute_positional_encoding(x)
        x = x * encoding.cos() + (self.rotate_half(x) * encoding.sin())

        return x


class Attention(nn.Module):
    """Attention.

    Example
    -------
    >>> module = Attention(
    ...    embedding_dimension=256,
    ...    heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        heads: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of heads.
        """

        super().__init__()

        self.heads = heads

        self.linear_1 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension * 3,
            bias=False,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

        self.rope = RoPE(
            embedding_dimension=embedding_dimension // heads,
            base=10_000,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        q, k, v = rearrange(self.linear_1(x), 'b s (n h e) -> n b h s e', n=3, h=self.heads)
        q, k = self.rope(q), self.rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = self.linear_2(rearrange(x, 'b h s e -> b s (h e)'))

        return x


class ResidualBlock(nn.Module):
    """Residual block.

    Example
    -------
    >>> module = Attention(
    ...     embedding_dimension=256,
    ...     heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of heads.
        """

        super().__init__()

        self.attention = Attention(
            embedding_dimension=embedding_dimension,
            heads=heads,
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * 3,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=embedding_dimension * 3,
                out_features=embedding_dimension,
            ),
        )

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_1(x))

        return x


class UpBlock(nn.Module):
    """Up block.

    Example
    -------
    >>> module = UpBlock(
    ...     embedding_dimension=256,
    ...     heads=16,
    ... )
    >>> x = torch.randn((1, 5, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of heads.
        """

        super().__init__()

        self.linear_upsample = LinearUpsample(
            embedding_dimension=embedding_dimension,
        )

        self.residual_block_1 = ResidualBlock(
            embedding_dimension=embedding_dimension,
            heads=heads,
        )

        self.residual_block_2 = ResidualBlock(
            embedding_dimension=embedding_dimension,
            heads=heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.linear_upsample(x)
        x = self.residual_block_1(x)
        x = self.residual_block_2(x)

        return x


class DownBlock(nn.Module):
    """Down block.

    Example
    -------
    >>> module = DownBlock(
    ...     embedding_dimension=256,
    ...     heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 5, 256).
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of heads.
        """

        super().__init__()

        self.linear_downsample = LinearDownsample(
            embedding_dimension=embedding_dimension,
        )

        self.residual_block_1 = ResidualBlock(
            embedding_dimension=embedding_dimension,
            heads=heads,
        )

        self.residual_block_2 = ResidualBlock(
            embedding_dimension=embedding_dimension,
            heads=heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.linear_downsample(x)
        x = self.residual_block_1(x)
        x = self.residual_block_2(x)

        return x


class Quantizer(nn.Module):
    """Quantizer.

    Example
    -------
    >>> module = Quantizer(
    ...     embedding_dimension=256,
    ...     quantizer_dimension=4,
    ...     quantizer_bits=5,
    ... )
    >>> x = torch.randn((1, 256, 10))
    >>> x = module.encode(x)  # Shape: (1, 4, 10).
    >>> x = module.decode(x)  # Shape: (1, 256, 10).
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        quantizer_dimension: int,
        quantizer_bits: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        latent_dimension : int
            The latent dimension.
        vocabulary_size : int
            The vocabulary size.
        """

        super().__init__()

        self.scale = (2 ** quantizer_bits) // 2

        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=embedding_dimension,
                out_features=quantizer_dimension,
            ),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=quantizer_dimension,
                out_features=embedding_dimension,
            ),
            nn.LeakyReLU(),
            nn.LayerNorm(normalized_shape=embedding_dimension),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.scale * self.encoder(x)
        x = x + (x.floor() - x).detach()

        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.decoder(x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        z : torch.Tensor
            The output latent tensor.
        """

        z = self.encode(x)
        x = self.decode(z)

        return x, z


@dataclass(frozen=True)
class AutoencoderConfiguration:
    input_dimension: int
    embedding_dimension: int
    quantizer_dimension: int
    quantizer_bits: int
    heads: int
    layers: int


class Autoencoder(nn.Module):
    """Autoencoder.

    Example
    -------
    >>> configuration = AutoencoderConfiguration(
    ...     input_dimension=3,
    ...     embedding_dimension=256,
    ...     quantizer_dimension=4,
    ...     quantizer_bits=5,
    ...     heads=16,
    ...     layers=3,
    ... )
    >>> module = Autoencoder(configuration=configuration)
    >>> x = torch.randn((1, 1024, 3))
    >>> z = module.encode(x)  # Shape: (1, 128, 4).
    >>> x = module.decode(z)  # Shape: (1, 1024, 3).
    """

    def __init__(self, *, configuration: AutoencoderConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : AutoencoderConfiguration
            The module configuration.
        """

        super().__init__()

        self.embedding = nn.Linear(
            in_features=configuration.input_dimension,
            out_features=configuration.embedding_dimension,
            bias=False,
        )

        self.unembedding = nn.Linear(
            in_features=configuration.embedding_dimension,
            out_features=configuration.input_dimension,
            bias=False,
        )

        self.encoder = nn.Sequential(*[
            DownBlock(
                embedding_dimension=configuration.embedding_dimension,
                heads=configuration.heads,
            ) for _ in range(configuration.layers)
        ])

        self.decoder = nn.Sequential(*[
            UpBlock(
                embedding_dimension=configuration.embedding_dimension,
                heads=configuration.heads,
            ) for _ in range(configuration.layers)
        ])

        self.quantizer = Quantizer(
            embedding_dimension=configuration.embedding_dimension,
            quantizer_dimension=configuration.quantizer_dimension,
            quantizer_bits=configuration.quantizer_bits,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.embedding(x)
        x = self.encoder(x)
        x = self.quantizer.encode(x)

        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.quantizer.decode(x)
        x = self.decoder(x)
        x = self.unembedding(x)
        x = torch.sigmoid(x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        z = self.encode(x)
        x = self.decode(z)

        return x, z
