from abc import ABC, abstractmethod
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import UniqueConstraint
from pydantic import (
    BaseModel,
    Field as PydanticField,
    PositiveFloat,
    computed_field
)

"""
****************************************************
User Agent Mapper classes
****************************************************
"""

class Application(SQLModel, table=True):
    """Applications identified by the mapper.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    vendor: str
    description: str
    user_agents: List["Mapping"] = Relationship(
        back_populates="application"
        )

    __table_args__ = (UniqueConstraint('name', name='uq_application_name'),)

class OperatingSystem(SQLModel, table=True):
    """Operating Systems identified by the mapper.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    user_agents: List["Mapping"] = Relationship(
        back_populates="operatingsystem"
        )

class Mapping(SQLModel, table=True):
    """Mapping objects that attribute a user agent string
    to an application and operating system.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_agent_string: str
    version: Optional[str]
    application: Application = Relationship(
        back_populates="user_agents"
        )
    app_id: int = Field(foreign_key="application.id")
    operatingsystem: Optional[OperatingSystem] = Relationship(
        back_populates="user_agents"
        )
    os_id: Optional[int] = Field(foreign_key="operatingsystem.id")

    def __repr__(self):
        """
        Return a human-readable string representation of this Mapping object.

        Returns:
            str: A formatted string including the user_agent_string, app_id, and os_id.
        """
        return f"Mapping({self.user_agent_string}, {self.app_id}, {self.os_id})"

class DataSource(BaseModel, ABC):
    """Abstract class for Mapper Data Sources, which map user agents to
    application names.
    """
    query_input: List[str] = PydanticField(default=[])
    query_time: PositiveFloat = PydanticField(default=0.0)
    hits: List[Mapping] = PydanticField(default=[])
    misses: List[str] = PydanticField(default=[])

    @computed_field
    @property
    def hits_found(self) -> bool:
        """
        Determine if any hits were found.

        Returns:
            bool: True if there is at least one Mapping in the hits list.
        """
        return len(self.hits) > 0
    
    @computed_field
    @property
    def misses_found(self) -> bool:
        """
        Determine if any misses were found.

        Returns:
            bool: True if there is at least one entry in the misses list.
        """
        return len(self.misses) > 0

class APIDataSource(DataSource, ABC):
    """Class for querying APIs to map user agents to their application info.
    """
    @abstractmethod
    def search(self) -> None:
        """
        Abstract method to search for user agent information using an API.

        This method must be implemented by any concrete subclass to
        perform the actual user-agent queries against a remote API.

        Raises:
            NotImplementedError: If called on the base class directly.
        """
        pass
