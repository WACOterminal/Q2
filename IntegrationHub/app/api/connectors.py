from fastapi import APIRouter, HTTPException, Depends
from typing import List

from ..models.connector import ConnectorMetadata
from shared.q_auth_parser.parser import get_current_user
from shared.q_auth_parser.models import UserClaims

router = APIRouter()

# In-memory database for demonstration purposes
# This would be populated from a registry, e.g., by scanning a directory or a database.
CONNECTORS_DB = {
    "slack-webhook": ConnectorMetadata(
        id="slack-webhook",
        name="Slack Webhook Notifier",
        version="1.0.0",
        description="Sends a message to a Slack channel via an incoming webhook.",
    ),
    "jira-create-issue": ConnectorMetadata(
        id="jira-create-issue",
        name="Jira: Create Issue",
        version="1.0.0",
        description="Creates a new issue in a Jira project.",
    ),
    "github": ConnectorMetadata(
        id="github",
        name="GitHub Integration",
        version="1.0.0",
        description="Comprehensive GitHub integration for repositories, issues, and pull requests.",
    ),
    "gitlab": ConnectorMetadata(
        id="gitlab",
        name="GitLab Integration",
        version="1.0.0",
        description="Complete GitLab integration for issues, merge requests, and CI/CD pipelines.",
    ),
    "nextcloud": ConnectorMetadata(
        id="nextcloud",
        name="NextCloud File Management",
        version="1.0.0",
        description="NextCloud integration for file management, sharing, and collaboration via WebDAV.",
    ),
    "onlyoffice": ConnectorMetadata(
        id="onlyoffice",
        name="OnlyOffice Document Server",
        version="1.0.0",
        description="OnlyOffice integration for collaborative document editing and management.",
    ),
    "openproject": ConnectorMetadata(
        id="openproject",
        name="OpenProject Management",
        version="1.0.0",
        description="OpenProject integration for work package and project management.",
    ),
    "kubernetes": ConnectorMetadata(
        id="kubernetes",
        name="Kubernetes Orchestration",
        version="1.0.0",
        description="Kubernetes integration for container orchestration and resource management.",
    ),
    "zulip": ConnectorMetadata(
        id="zulip",
        name="Zulip Chat Integration",
        version="1.0.0",
        description="Zulip chat platform integration for team communication.",
    ),
    "smtp-email": ConnectorMetadata(
        id="smtp-email",
        name="SMTP Email Service",
        version="1.0.0",
        description="SMTP email integration for sending automated notifications.",
    ),
    "http": ConnectorMetadata(
        id="http",
        name="HTTP Generic Connector",
        version="1.0.0",
        description="Generic HTTP connector for integrating with any REST API.",
    ),
    "pulsar-publish": ConnectorMetadata(
        id="pulsar-publish",
        name="Apache Pulsar Publisher",
        version="1.0.0",
        description="Apache Pulsar integration for event streaming and messaging.",
    ),
    "freecad": ConnectorMetadata(
        id="freecad",
        name="FreeCAD CAD Software",
        version="1.0.0",
        description="FreeCAD integration for CAD file management, 3D modeling, and automated design workflows.",
    ),
    "multimedia": ConnectorMetadata(
        id="multimedia",
        name="Multimedia Processing Suite",
        version="1.0.0",
        description="Multimedia processing integration for Audacity, GIMP, and OpenShot automation.",
    ),
}

@router.get("/", response_model=List[ConnectorMetadata])
def list_connectors(user: UserClaims = Depends(get_current_user)):
    """
    List all available connectors in the marketplace.
    """
    return list(CONNECTORS_DB.values())

@router.get("/{connector_id}", response_model=ConnectorMetadata)
def get_connector(connector_id: str, user: UserClaims = Depends(get_current_user)):
    """
    Retrieve a single connector's details by its ID.
    """
    if connector_id not in CONNECTORS_DB:
        raise HTTPException(status_code=404, detail=f"Connector with ID {connector_id} not found")
    return CONNECTORS_DB[connector_id] 