from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Dict, Any
from pydantic import BaseModel

from app.core.engine import engine
from app.core.pulsar_client import publish_event
from shared.q_auth_parser.parser import get_current_user
from shared.q_auth_parser.models import UserClaims

router = APIRouter()

# --- Pre-defined Flows ---
# In a real system, these would be stored in a database.
PREDEFINED_FLOWS: Dict[str, Dict[str, Any]] = {
    "send-summary-email": {
        "id": "send-summary-email",
        "name": "Send Summary Email via SMTP",
        "description": "A flow to send an email. Requires 'to', 'subject', and 'body' in the parameters.",
        "steps": [
            {
                "name": "Send Email",
                "connector_id": "smtp-email",
                "action_id": "send",
                # This credential must be created in Vault beforehand
                "credential_id": "smtp-credentials", 
                # The 'configuration' here is filled by the trigger parameters
            }
        ]
    },
    "ingest-zulip-to-kg": {
        "id": "ingest-zulip-to-kg",
        "name": "Ingest Zulip Stream to Knowledge Graph",
        "description": "Fetches recent messages from a Zulip stream and publishes them to a Pulsar topic for ingestion into the Knowledge Graph.",
        "steps": [
            {
                "name": "Fetch Messages from Zulip",
                "connector_id": "zulip",
                "action_id": "get-messages",
                "credential_id": "zulip-credentials",
                "configuration": {
                    "stream": "knowledge-graph",
                    "num_before": 50
                }
            },
            {
                "name": "Publish to Pulsar for KG Ingestion",
                "connector_id": "pulsar-publish",
                "action_id": "default_action", # Not strictly needed, but good for clarity
                "configuration": {
                    "topic": "persistent://public/default/knowledge-graph-ingestion",
                    # The 'message' will be the output from the previous step
                    # The engine now maps this automatically.
                }
            }
        ]
    },
    "post_daily_zulip_summary": {
        "id": "post_daily_zulip_summary",
        "name": "Post Daily Zulip Summary",
        "description": "Asks an agent to summarize a Zulip stream and posts the result to another stream. A proactive, scheduled task.",
        "steps": [
            {
                "name": "Ask Agent for Summary",
                "connector_id": "http",
                "credential_id": "managerq-service-token", # A service account token for managerQ
                "configuration": {
                    "method": "POST",
                    "url": "http://managerq:8003/v1/tasks",
                    "json": {
                        "prompt": "Using the summarize_stream_activity tool, create a summary for the 'knowledge-graph' stream for the past 24 hours. In your final answer, include ONLY the summary text itself, without any conversational pleasantries."
                    }
                }
            },
            {
                "name": "Post Summary to Zulip",
                "connector_id": "zulip",
                "action_id": "send-message",
                "credential_id": "zulip-credentials",
                "configuration": {
                    "stream": "daily-digest",
                    "topic": "Daily Summary for {{ 'now' | date:'%Y-%m-%d' }}", # Simple templating could be added
                    "content": "Good morning! Here is the summary of yesterday's activity in the #knowledge-graph stream:\n\n> {{ result }}"
                }
            }
        ]
    },
    "post_comment_on_pr": {
        "id": "post_comment_on_pr",
        "name": "Post a Comment on a GitHub PR",
        "description": "Uses the GitHub connector to post a comment on a specified pull request.",
        "steps": [
            {
                "name": "Create PR Comment",
                "connector_id": "github",
                "action_id": "create_pull_request_comment",
                "credential_id": "github-pat", # A credential containing the GitHub PAT
                "configuration": {
                    # These values will be provided by the flow trigger
                    "repo": "{{ repo }}",
                    "pr_number": "{{ pr_number }}",
                    "body": "{{ body }}"
                }
            }
        ]
    },
    "code_review_agent": {
        "id": "code_review_agent",
        "name": "Code Review Agent",
        "description": "Triggered by a GitHub webhook on PR creation. Asks an agent to review the code changes and posts the feedback as a comment.",
        "steps": [
            {
                "name": "Get PR Diff",
                "connector_id": "github",
                "action_id": "get_pr_diff",
                "credential_id": "github-pat",
                "configuration": {
                    "repo": "{{ repo }}",
                    "pr_number": "{{ pr_number }}"
                }
            },
            {
                "name": "Ask Agent for Review",
                "connector_id": "http",
                "credential_id": "managerq-service-token",
                "configuration": {
                    "method": "POST",
                    "url": "http://managerq:8003/v1/tasks",
                    "json": {
                        "prompt": "You are a senior software engineer performing a code review. Please analyze the following diff and provide a high-level summary of the changes, identify potential issues, and offer constructive feedback. Structure your response in clear markdown format. Here is the diff:\n\n```diff\n{{ diff }}\n```"
                    }
                }
            },
            {
                "name": "Post Review Comment to PR",
                "connector_id": "github",
                "action_id": "create_pull_request_comment",
                "credential_id": "github-pat",
                "configuration": {
                    "repo": "{{ repo }}",
                    "pr_number": "{{ pr_number }}",
                    "body": "### AI Code Review\n\nHere is a summary of my review:\n\n{{ result }}"
                }
            }
        ]
    },
    "propose_code_fix": {
        "id": "propose_code_fix",
        "name": "Propose a Code Fix via Pull Request",
        "description": "Creates a new branch, commits a single file change, and opens a pull request for review. Ideal for agent-driven code remediation.",
        "steps": [
            {
                "name": "Create New Branch",
                "connector_id": "github",
                "action_id": "create_branch",
                "credential_id": "github-pat",
                "configuration": {
                    "repo": "{{ repo }}",
                    "new_branch_name": "{{ new_branch_name }}",
                    "source_branch_name": "{{ source_branch_name }}"
                }
            },
            {
                "name": "Commit File Change",
                "connector_id": "github",
                "action_id": "create_commit",
                "credential_id": "github-pat",
                "configuration": {
                    "repo": "{{ repo }}",
                    "branch_name": "{{ new_branch_name }}",
                    "file_path": "{{ file_path }}",
                    "new_content": "{{ new_content }}",
                    "commit_message": "{{ commit_message }}"
                }
            },
            {
                "name": "Create Pull Request",
                "connector_id": "github",
                "action_id": "create_pull_request",
                "credential_id": "github-pat",
                "configuration": {
                    "repo": "{{ repo }}",
                    "head_branch": "{{ new_branch_name }}",
                    "base_branch": "{{ source_branch_name }}",
                    "title": "{{ pr_title }}",
                    "body": "{{ pr_body }}"
                }
            }
        ]
    },
    "get_pr_diff": {
        "id": "get_pr_diff",
        "name": "Get PR Diff",
        "description": "Fetches the raw diff of a pull request.",
        "steps": [
            {
                "name": "Get PR Diff",
                "connector_id": "github",
                "action_id": "get_pr_diff",
                "credential_id": "github-pat",
                "configuration": {
                    "repo": "{{ repo }}",
                    "pr_number": "{{ pr_number }}"
                }
            }
        ]
    },
    "create-jira-issue": {
        "id": "create-jira-issue",
        "name": "Create Jira Issue",
        "description": "Creates a new issue in a Jira project.",
        "trigger": {
            "type": "manual",
            "configuration": {
                "parameters": ["project_key", "summary", "description", "issue_type"]
            }
        },
        "steps": [
            {
                "name": "create_issue",
                "connector_id": "http",
                "credential_id": "jira-credentials", # Expects a secret with 'username' and 'api_token'
                "configuration": {
                    "action_id": "post",
                    # The jira_url would come from a global config, but is hardcoded here for example
                    "url": "https://your-jira-instance.atlassian.net/rest/api/2/issue",
                    "auth_method": "basic", # The http connector would need to support this
                    "json_template": {
                        "fields": {
                            "project": {
                                "key": "{{ trigger.project_key }}"
                            },
                            "summary": "{{ trigger.summary }}",
                            "description": "{{ trigger.description }}",
                            "issuetype": {
                                "name": "{{ trigger.issue_type | default('Task') }}"
                            }
                        }
                    }
                },
                "dependencies": []
            }
        ]
    },
    "ingest-github-issues-to-search": {
        "id": "ingest-github-issues-to-search",
        "name": "Ingest GitHub Issues to Cognitive Search",
        "description": "Fetches recent issues from a GitHub repo and ingests them into VectorStoreQ and KnowledgeGraphQ.",
        "trigger": {
            "type": "manual",
            "configuration": {
                "parameters": ["repo_owner", "repo_name"]
            }
        },
        "steps": [
            {
                "name": "fetch_issues",
                "connector_id": "github",
                "credential_id": "github-pat",
                "configuration": {
                    "action_id": "get_issues",
                    "repo": "{{ trigger.repo_owner }}/{{ trigger.repo_name }}",
                    "state": "open"
                },
                "dependencies": []
            },
            {
                "name": "ingest_to_vectorstore",
                "connector_id": "http", # Assuming a direct HTTP call to VectorStoreQ
                "credential_id": "vectorstoreq-service-token",
                "dependencies": ["fetch_issues"],
                "configuration": {
                    "action_id": "ingest_batch",
                    "method": "POST",
                    "url": "http://vectorstoreq:8000/v1/ingest/batch",
                    "json_template": {
                        "collection_name": "github_issues",
                        "documents": "{{ fetch_issues.result | map(attribute='body') | list }}",
                        "metadatas": "{{ fetch_issues.result }}"
                    }
                }
            },
            {
                "name": "ingest_to_knowledgegraph",
                "connector_id": "http", # Assuming a direct HTTP call to KnowledgeGraphQ
                "credential_id": "knowledgegraphq-service-token",
                "dependencies": ["fetch_issues"],
                "configuration": {
                    "action_id": "ingest_batch",
                    "method": "POST",
                    "url": "http://knowledgegraphq:8000/v1/ingest/batch",
                    "json_template": {
                        "source": "github",
                        "type": "issue",
                        "records": "{{ fetch_issues.result }}"
                    }
                }
            }
        ]
    },
    "create-gitlab-issue": {
        "id": "create-gitlab-issue",
        "name": "Create GitLab Issue",
        "description": "Creates a new issue in a GitLab project.",
        "trigger": {
            "type": "manual",
            "configuration": {
                "parameters": ["project_id", "title", "description", "labels"]
            }
        },
        "steps": [
            {
                "name": "create_issue",
                "connector_id": "gitlab",
                "credential_id": "gitlab-credentials",
                "configuration": {
                    "action_id": "create_issue",
                    "project_id": "{{ trigger.project_id }}"
                },
                "dependencies": []
            }
        ]
    },
    "gitlab-mr-workflow": {
        "id": "gitlab-mr-workflow",
        "name": "GitLab Merge Request Workflow",
        "description": "Creates a branch, commits changes, and opens a merge request in GitLab.",
        "trigger": {
            "type": "manual",
            "configuration": {
                "parameters": ["project_id", "branch_name", "title", "description", "file_path", "content", "commit_message"]
            }
        },
        "steps": [
            {
                "name": "create_branch",
                "connector_id": "gitlab",
                "credential_id": "gitlab-credentials",
                "configuration": {
                    "action_id": "create_branch",
                    "project_id": "{{ trigger.project_id }}"
                },
                "dependencies": []
            },
            {
                "name": "create_commit",
                "connector_id": "gitlab",
                "credential_id": "gitlab-credentials",
                "configuration": {
                    "action_id": "create_commit",
                    "project_id": "{{ trigger.project_id }}"
                },
                "dependencies": ["create_branch"]
            },
            {
                "name": "create_merge_request",
                "connector_id": "gitlab",
                "credential_id": "gitlab-credentials",
                "configuration": {
                    "action_id": "create_merge_request",
                    "project_id": "{{ trigger.project_id }}"
                },
                "dependencies": ["create_commit"]
            }
        ]
    },
    "nextcloud-document-workflow": {
        "id": "nextcloud-document-workflow",
        "name": "NextCloud Document Workflow",
        "description": "Uploads a document to NextCloud and creates a share link.",
        "trigger": {
            "type": "manual",
            "configuration": {
                "parameters": ["file_path", "content", "share_type"]
            }
        },
        "steps": [
            {
                "name": "upload_file",
                "connector_id": "nextcloud",
                "credential_id": "nextcloud-credentials",
                "configuration": {
                    "action_id": "upload_file",
                    "file_path": "{{ trigger.file_path }}"
                },
                "dependencies": []
            },
            {
                "name": "create_share",
                "connector_id": "nextcloud",
                "credential_id": "nextcloud-credentials",
                "configuration": {
                    "action_id": "share_file",
                    "file_path": "{{ trigger.file_path }}",
                    "share_type": "{{ trigger.share_type | default(3) }}"
                },
                "dependencies": ["upload_file"]
            }
        ]
    },
    "onlyoffice-collaborative-editing": {
        "id": "onlyoffice-collaborative-editing",
        "name": "OnlyOffice Collaborative Editing Setup",
        "description": "Creates a document in OnlyOffice and shares it with specified users.",
        "trigger": {
            "type": "manual",
            "configuration": {
                "parameters": ["title", "document_type", "users", "permissions"]
            }
        },
        "steps": [
            {
                "name": "create_document",
                "connector_id": "onlyoffice",
                "credential_id": "onlyoffice-credentials",
                "configuration": {
                    "action_id": "create_document",
                    "document_type": "{{ trigger.document_type | default('docx') }}"
                },
                "dependencies": []
            },
            {
                "name": "share_document",
                "connector_id": "onlyoffice",
                "credential_id": "onlyoffice-credentials",
                "configuration": {
                    "action_id": "share_document",
                    "permissions": "{{ trigger.permissions | default('write') }}"
                },
                "dependencies": ["create_document"]
            }
        ]
    },
    "project-workflow-automation": {
        "id": "project-workflow-automation",
        "name": "Project Workflow Automation",
        "description": "Comprehensive workflow that creates a GitLab issue, shares documents via NextCloud, and sets up collaborative editing in OnlyOffice.",
        "trigger": {
            "type": "manual",
            "configuration": {
                "parameters": ["project_id", "issue_title", "issue_description", "document_title", "collaborators"]
            }
        },
        "steps": [
            {
                "name": "create_gitlab_issue",
                "connector_id": "gitlab",
                "credential_id": "gitlab-credentials",
                "configuration": {
                    "action_id": "create_issue",
                    "project_id": "{{ trigger.project_id }}"
                },
                "dependencies": []
            },
            {
                "name": "create_project_folder",
                "connector_id": "nextcloud",
                "credential_id": "nextcloud-credentials",
                "configuration": {
                    "action_id": "create_folder",
                    "folder_path": "projects/{{ trigger.project_id }}-{{ trigger.issue_title | slugify }}"
                },
                "dependencies": []
            },
            {
                "name": "create_collaborative_document",
                "connector_id": "onlyoffice",
                "credential_id": "onlyoffice-credentials",
                "configuration": {
                    "action_id": "create_document",
                    "document_type": "docx"
                },
                "dependencies": ["create_project_folder"]
            },
            {
                "name": "share_document_with_team",
                "connector_id": "onlyoffice",
                "credential_id": "onlyoffice-credentials",
                "configuration": {
                    "action_id": "share_document",
                    "permissions": "write"
                },
                "dependencies": ["create_collaborative_document"]
            },
            {
                "name": "comment_on_gitlab_issue",
                "connector_id": "gitlab",
                "credential_id": "gitlab-credentials",
                "configuration": {
                    "action_id": "create_comment",
                    "project_id": "{{ trigger.project_id }}",
                    "resource_type": "issues"
                },
                "dependencies": ["create_gitlab_issue", "share_document_with_team"]
            }
        ]
    },
    "digital-content-pipeline": {
        "id": "digital-content-pipeline",
        "name": "Digital Content Creation Pipeline",
        "description": "Complete digital content pipeline: design in FreeCAD, process images in GIMP, edit videos in OpenShot, and share via NextCloud.",
        "trigger": {
            "type": "manual",
            "configuration": {
                "parameters": ["project_name", "cad_specifications", "image_assets", "video_requirements"]
            }
        },
        "steps": [
            {
                "name": "create_cad_model",
                "connector_id": "freecad",
                "credential_id": "freecad-credentials",
                "configuration": {
                    "action_id": "create_document"
                },
                "dependencies": []
            },
            {
                "name": "generate_renderings",
                "connector_id": "freecad",
                "credential_id": "freecad-credentials",
                "configuration": {
                    "action_id": "export_document",
                    "export_format": "STL"
                },
                "dependencies": ["create_cad_model"]
            },
            {
                "name": "process_images",
                "connector_id": "multimedia",
                "credential_id": "multimedia-credentials",
                "configuration": {
                    "action_id": "convert_image"
                },
                "dependencies": ["generate_renderings"]
            },
            {
                "name": "create_marketing_video",
                "connector_id": "multimedia",
                "credential_id": "multimedia-credentials",
                "configuration": {
                    "action_id": "export_video"
                },
                "dependencies": ["process_images"]
            },
            {
                "name": "upload_to_nextcloud",
                "connector_id": "nextcloud",
                "credential_id": "nextcloud-credentials",
                "configuration": {
                    "action_id": "create_folder",
                    "folder_path": "projects/{{ trigger.project_name }}/deliverables"
                },
                "dependencies": ["create_marketing_video"]
            },
            {
                "name": "share_with_stakeholders",
                "connector_id": "nextcloud",
                "credential_id": "nextcloud-credentials",
                "configuration": {
                    "action_id": "share_file",
                    "share_type": 3
                },
                "dependencies": ["upload_to_nextcloud"]
            }
        ]
    },
    "automated-design-review": {
        "id": "automated-design-review",
        "name": "Automated Design Review Process",
        "description": "Automated workflow that reviews CAD designs, generates technical documentation, and creates GitLab issues for feedback.",
        "trigger": {
            "type": "manual",
            "configuration": {
                "parameters": ["cad_file_path", "project_id", "review_checklist"]
            }
        },
        "steps": [
            {
                "name": "analyze_cad_model",
                "connector_id": "freecad",
                "credential_id": "freecad-credentials",
                "configuration": {
                    "action_id": "calculate_volume"
                },
                "dependencies": []
            },
            {
                "name": "generate_technical_drawings",
                "connector_id": "freecad",
                "credential_id": "freecad-credentials",
                "configuration": {
                    "action_id": "generate_technical_drawing"
                },
                "dependencies": ["analyze_cad_model"]
            },
            {
                "name": "create_review_document",
                "connector_id": "onlyoffice",
                "credential_id": "onlyoffice-credentials",
                "configuration": {
                    "action_id": "create_document",
                    "document_type": "docx"
                },
                "dependencies": ["generate_technical_drawings"]
            },
            {
                "name": "create_gitlab_issue",
                "connector_id": "gitlab",
                "credential_id": "gitlab-credentials",
                "configuration": {
                    "action_id": "create_issue",
                    "project_id": "{{ trigger.project_id }}"
                },
                "dependencies": ["create_review_document"]
            },
            {
                "name": "notify_team",
                "connector_id": "smtp-email",
                "credential_id": "smtp-credentials",
                "configuration": {
                    "action_id": "send"
                },
                "dependencies": ["create_gitlab_issue"]
            }
        ]
    },
    "multimedia-content-automation": {
        "id": "multimedia-content-automation",
        "name": "Multimedia Content Automation",
        "description": "Automated multimedia processing: batch process images, enhance audio, create video montages, and distribute content.",
        "trigger": {
            "type": "manual",
            "configuration": {
                "parameters": ["source_folder", "output_specifications", "distribution_channels"]
            }
        },
        "steps": [
            {
                "name": "batch_process_images",
                "connector_id": "multimedia",
                "credential_id": "multimedia-credentials",
                "configuration": {
                    "action_id": "batch_process"
                },
                "dependencies": []
            },
            {
                "name": "enhance_audio",
                "connector_id": "multimedia",
                "credential_id": "multimedia-credentials",
                "configuration": {
                    "action_id": "apply_effects"
                },
                "dependencies": []
            },
            {
                "name": "create_video_montage",
                "connector_id": "multimedia",
                "credential_id": "multimedia-credentials",
                "configuration": {
                    "action_id": "export_video"
                },
                "dependencies": ["batch_process_images", "enhance_audio"]
            },
            {
                "name": "upload_to_storage",
                "connector_id": "nextcloud",
                "credential_id": "nextcloud-credentials",
                "configuration": {
                    "action_id": "upload_file"
                },
                "dependencies": ["create_video_montage"]
            },
            {
                "name": "create_sharing_links",
                "connector_id": "nextcloud",
                "credential_id": "nextcloud-credentials",
                "configuration": {
                    "action_id": "share_file"
                },
                "dependencies": ["upload_to_storage"]
            }
        ]
    },
    "engineering-workflow-complete": {
        "id": "engineering-workflow-complete",
        "name": "Complete Engineering Workflow",
        "description": "End-to-end engineering workflow: requirements in OpenProject, design in FreeCAD, code in GitLab, documentation in OnlyOffice, and deliverables in NextCloud.",
        "trigger": {
            "type": "manual",
            "configuration": {
                "parameters": ["project_id", "requirements", "design_specs", "timeline"]
            }
        },
        "steps": [
            {
                "name": "create_work_package",
                "connector_id": "openproject",
                "credential_id": "openproject-credentials",
                "configuration": {
                    "action_id": "create_work_package"
                },
                "dependencies": []
            },
            {
                "name": "create_cad_design",
                "connector_id": "freecad",
                "credential_id": "freecad-credentials",
                "configuration": {
                    "action_id": "create_document"
                },
                "dependencies": ["create_work_package"]
            },
            {
                "name": "setup_code_repository",
                "connector_id": "gitlab",
                "credential_id": "gitlab-credentials",
                "configuration": {
                    "action_id": "create_branch",
                    "project_id": "{{ trigger.project_id }}"
                },
                "dependencies": ["create_cad_design"]
            },
            {
                "name": "create_technical_documentation",
                "connector_id": "onlyoffice",
                "credential_id": "onlyoffice-credentials",
                "configuration": {
                    "action_id": "create_document",
                    "document_type": "docx"
                },
                "dependencies": ["setup_code_repository"]
            },
            {
                "name": "organize_project_deliverables",
                "connector_id": "nextcloud",
                "credential_id": "nextcloud-credentials",
                "configuration": {
                    "action_id": "create_folder",
                    "folder_path": "engineering-projects/{{ trigger.project_id }}"
                },
                "dependencies": ["create_technical_documentation"]
            },
            {
                "name": "share_with_stakeholders",
                "connector_id": "nextcloud",
                "credential_id": "nextcloud-credentials",
                "configuration": {
                    "action_id": "share_file"
                },
                "dependencies": ["organize_project_deliverables"]
            },
            {
                "name": "update_project_status",
                "connector_id": "openproject",
                "credential_id": "openproject-credentials",
                "configuration": {
                    "action_id": "update_work_package"
                },
                "dependencies": ["share_with_stakeholders"]
            }
        ]
    }
}


class Flow(BaseModel):
    id: str
    name: str
    description: str

class TriggerRequest(BaseModel):
    parameters: Dict[str, Any]


@router.get("", response_model=List[Flow])
async def list_flows(user: UserClaims = Depends(get_current_user)):
    """Lists all available pre-defined flows."""
    return [Flow(**flow) for flow in PREDEFINED_FLOWS.values()]

@router.post("/{flow_id}/trigger")
async def trigger_flow(
    flow_id: str,
    request: TriggerRequest,
    user: UserClaims = Depends(get_current_user)
):
    """Triggers a pre-defined flow by its ID."""
    if flow_id not in PREDEFINED_FLOWS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Flow not found.")
    
    # Publish an event about the trigger
    await publish_event(
        event_type="flow.triggered",
        source="IntegrationHub",
        payload={
            "flow_id": flow_id,
            "user": user.dict(),
            "trigger_params": request.parameters
        }
    )

    flow_config = PREDEFINED_FLOWS[flow_id]
    
    # The new engine handles parameter mapping internally.
    # The initial trigger parameters are passed as the starting data_context.
    await engine.run_flow(flow_config, data_context=request.parameters)
    
    return {"status": "Flow triggered successfully", "flow_id": flow_id} 