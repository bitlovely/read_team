import os
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator, model_validator, AnyUrl

from .common import generate_constants_docs


class Constants(BaseModel):
    """
    Configuration constants for the application.
    """

    # Environment settings
    TESTNET: bool = Field(
        default_factory=lambda: os.getenv("TESTNET", "0").strip().lower()
        in ("1", "true", "yes"),
        description="Flag to indicate if running in testnet mode.",
    )

    # Versioning
    VERSION: str = Field(
        default="0.0.1",
        description="Version of the application in 'major.minor.patch' format.",
    )
    SPEC_VERSION: int = Field(
        default=0,
        description="Specification version calculated from the version string.",
    )

    # Challenge settings
    N_CHALLENGES_PER_EPOCH: int = Field(
        default=10, description="Number of challenges per epoch."
    )
    SCORING_HOUR: int = Field(
        default=14, description="Hour of the day when scoring occurs (0-23)."
    )
    POINT_DECAY_RATE: float = Field(
        default=1 / 14, description="Daily point decay rate."
    )

    # Network settings
    CHALLENGE_DOCKER_PORT: int = Field(
        default=10001, description="Port used for challenge Docker containers."
    )
    MINER_DOCKER_PORT: int = Field(
        default=10002, description="Port used for miner Docker containers."
    )

    # Time intervals (in seconds)
    REVEAL_INTERVAL: int = Field(
        default=3600 * 24, description="Time interval for revealing commits."
    )
    EPOCH_LENGTH: int = Field(
        default=3600, description="Length of an epoch in seconds."
    )
    MIN_VALIDATOR_STAKE: int = Field(
        default=10_000, description="Minimum validator stake required."
    )

    # Query settings
    QUERY_TIMEOUT: int = Field(default=30, description="Timeout for queries in seconds.")

    # Centralized API settings
    STORAGE_URL: AnyUrl = Field(
        default="http://storage.redteam.technology/storage",
        description="URL for storing miners' work"
    )
    REWARDING_URL: AnyUrl = Field(
        default="http://storage.redteam.technology/rewarding",
        description="URL for rewarding miners"
    )

    class Config:
        validate_assignment = True

    @field_validator("SPEC_VERSION", mode="before")
    def calculate_spec_version(cls, v, values):
        """
        Calculates the specification version as an integer based on the version string.

        Args:
            v: The current value of spec_version (unused).
            values: Dictionary of field values.

        Returns:
            int: The calculated specification version.
        """
        version_str = values.get("VERSION", "0.0.1")
        try:
            major, minor, patch = (int(part) for part in version_str.split("."))
            return (1000 * major) + (10 * minor) + patch
        except ValueError as e:
            raise ValueError(
                f"Invalid version format '{version_str}'. Expected 'major.minor.patch'."
            ) from e

    @model_validator(mode="before")
    def adjust_for_testnet(cls, values):
        """
        Adjusts certain constants based on whether TESTNET mode is enabled.

        Args:
            values: Dictionary of field values.

        Returns:
            dict: The adjusted values dictionary.
        """
        testnet = os.environ.get("TESTNET", "")
        is_testnet = testnet.lower() in ("1", "true", "yes")
        print(f"Testnet mode: {is_testnet}, {testnet}")
        if is_testnet:
            print("Adjusting constants for testnet mode.")
            values["REVEAL_INTERVAL"] = 30
            values["EPOCH_LENGTH"] = 30
            values["MIN_VALIDATOR_STAKE"] = -1
        return values

    def decay_points(self, point: float, days_passed: int) -> float:
        """
        Applies decay to the given points based on the number of days passed.

        Args:
            point (float): The original point value.
            days_passed (int): The number of days since the point was awarded.

        Returns:
            float: The decayed point value.
        """
        decay_factor = 1 - min(self.POINT_DECAY_RATE * days_passed, 1)
        return point * decay_factor

    def is_commit_on_time(self, commit_timestamp: float) -> bool:
        """
        Validator do scoring every day at SCORING_HOUR.
        So the commit time should be submitted before the previous day's SCORING_HOUR.
        """
        today_closed_time = datetime.now().replace(
            hour=self.SCORING_HOUR, minute=0, second=0, microsecond=0
        )
        previous_day_closed_time = today_closed_time - timedelta(days=1)
        return commit_timestamp < previous_day_closed_time.timestamp()


constants = Constants()


if __name__ == "__main__":
    from termcolor import colored

    def print_with_colors(content: str):
        """
        Prints the content with basic colors using termcolor.

        Args:
            content (str): The content to print.
        """
        print(colored(content, "cyan"))

    markdown_content = generate_constants_docs(Constants)

    print_with_colors(markdown_content)
