"""Width and depth sweep management for experiments."""

from dataclasses import dataclass
from typing import Iterator, List, Optional

from ..models.config import ModelConfiguration
from ..data.types import SweepType


@dataclass
class SweepPoint:
    """A single point in the capacity sweep.

    Attributes:
        width: Network width.
        depth: Network depth.
        config: Full model configuration for this point.
        index: Position in the sweep order.
    """

    width: int
    depth: int
    config: ModelConfiguration
    index: int

    @property
    def run_id(self) -> str:
        """Generate a run ID for this sweep point."""
        return f"width_{self.width:04d}_depth_{self.depth:02d}"

    @property
    def n_parameters(self, n_features: int = 20) -> int:
        """Estimate number of parameters."""
        return self.config.n_parameters(n_features)


class CapacitySweep:
    """Manager for capacity sweep configurations.

    Generates model configurations for width or depth sweeps,
    tracks completed points, and supports resumption.

    Args:
        widths: List of widths to sweep.
        depths: List of depths to sweep.
        base_config: Base model configuration.
        sweep_type: Type of sweep (WIDTH or DEPTH).
    """

    def __init__(
        self,
        widths: List[int],
        depths: List[int],
        base_config: ModelConfiguration,
        sweep_type: SweepType = SweepType.WIDTH,
    ):
        self.widths = sorted(widths)
        self.depths = sorted(depths)
        self.base_config = base_config
        self.sweep_type = sweep_type

        # Generate all sweep points
        self.points: List[SweepPoint] = []
        self._generate_points()

        # Track completion status
        self.completed: set = set()

    def _generate_points(self) -> None:
        """Generate all sweep points based on sweep type."""
        index = 0

        if self.sweep_type == SweepType.WIDTH:
            # Primary sweep over width, secondary over depth
            for width in self.widths:
                for depth in self.depths:
                    config = self.base_config.with_width(width).with_depth(depth)
                    point = SweepPoint(
                        width=width,
                        depth=depth,
                        config=config,
                        index=index,
                    )
                    self.points.append(point)
                    index += 1
        else:
            # Primary sweep over depth, secondary over width
            for depth in self.depths:
                for width in self.widths:
                    config = self.base_config.with_depth(depth).with_width(width)
                    point = SweepPoint(
                        width=width,
                        depth=depth,
                        config=config,
                        index=index,
                    )
                    self.points.append(point)
                    index += 1

    def __len__(self) -> int:
        """Total number of sweep points."""
        return len(self.points)

    def __iter__(self) -> Iterator[SweepPoint]:
        """Iterate over all sweep points."""
        return iter(self.points)

    def mark_completed(self, point: SweepPoint) -> None:
        """Mark a sweep point as completed.

        Args:
            point: The completed sweep point.
        """
        self.completed.add(point.run_id)

    def mark_completed_by_id(self, run_id: str) -> None:
        """Mark a sweep point as completed by run ID.

        Args:
            run_id: The run ID to mark as completed.
        """
        self.completed.add(run_id)

    def is_completed(self, point: SweepPoint) -> bool:
        """Check if a sweep point is completed.

        Args:
            point: The sweep point to check.

        Returns:
            True if completed.
        """
        return point.run_id in self.completed

    def get_pending(self) -> List[SweepPoint]:
        """Get list of pending (not completed) sweep points.

        Returns:
            List of pending sweep points in order.
        """
        return [p for p in self.points if not self.is_completed(p)]

    def get_next(self) -> Optional[SweepPoint]:
        """Get the next pending sweep point.

        Returns:
            Next pending point, or None if all completed.
        """
        pending = self.get_pending()
        return pending[0] if pending else None

    @property
    def n_completed(self) -> int:
        """Number of completed sweep points."""
        return len(self.completed)

    @property
    def n_pending(self) -> int:
        """Number of pending sweep points."""
        return len(self) - self.n_completed

    @property
    def progress_fraction(self) -> float:
        """Completion progress as a fraction."""
        if len(self) == 0:
            return 1.0
        return self.n_completed / len(self)

    def progress_string(self) -> str:
        """Human-readable progress string."""
        return f"{self.n_completed}/{len(self)} ({self.progress_fraction:.1%})"
