# This file contains the Istio and Vault security configurations for the Q Platform.

# --- Istio Service Mesh Security ---

# 1. RequestAuthentication: Tells the ingress gateway how to validate JWTs.
resource "kubernetes_manifest" "jwt_validator" {
  manifest = {
    "apiVersion" = "security.istio.io/v1"
    "kind"       = "RequestAuthentication"
    "metadata" = {
      "name"      = "jwt-validator"
      # This policy applies to the istio-system namespace where the gateway lives
      "namespace" = "istio-system"
    }
    "spec" = {
      "selector" = {
        "matchLabels" = {
          "istio" = "ingressgateway"
        }
      }
      "jwtRules" = [
        {
          "issuer"               = var.keycloak_issuer_url
          "jwksUri"              = "${var.keycloak_issuer_url}/protocol/openid-connect/certs"
          "outputPayloadToHeader" = "X-User-Claims"
          "forwardOriginalToken" = true
        }
      ]
    }
  }
  depends_on = [helm_release.keycloak, helm_release.istiod]
}

# 2. AuthorizationPolicy: Enforces that a valid JWT is required for all requests.
resource "kubernetes_manifest" "require_jwt" {
  manifest = {
    "apiVersion" = "security.istio.io/v1"
    "kind"       = "AuthorizationPolicy"
    "metadata" = {
      "name"      = "require-jwt-for-q-platform"
      # This policy applies to the namespace where our services are deployed
      "namespace" = var.namespace
    }
    "spec" = {
      # By default, this applies to all workloads in the namespace
      "action" = "ALLOW"
      "rules" = [
        {
          "from" = [
            {
              "source" = {
                # This requires that the request principal is not empty,
                # which means the JWT was successfully validated by the
                # RequestAuthentication policy.
                "requestPrincipals" = ["*"]
              }
            }
          ]
        }
      ]
    }
  }
  depends_on = [helm_release.keycloak, helm_release.istiod]
}

# 3. PeerAuthentication: Enable mTLS for all services in the namespace
resource "kubernetes_manifest" "mtls_policy" {
  manifest = {
    "apiVersion" = "security.istio.io/v1"
    "kind"       = "PeerAuthentication"
    "metadata" = {
      "name"      = "default-mtls"
      "namespace" = var.namespace
    }
    "spec" = {
      "mtls" = {
        "mode" = "STRICT"
      }
    }
  }
  depends_on = [helm_release.istiod]
}

# --- Vault Kubernetes Integration ---

# 1. Enable the Kubernetes auth method in Vault.
resource "vault_auth_backend" "kubernetes" {
  type = "kubernetes"
  path = "kubernetes"
  depends_on = [helm_release.vault]
}

# 2. Configure the Kubernetes auth method.
resource "vault_kubernetes_auth_backend_config" "k8s_config" {
  backend         = vault_auth_backend.kubernetes.path
  kubernetes_host = "https://kubernetes.default.svc"
  depends_on = [helm_release.vault]
}

# 3. Create secrets engine for Q Platform services
resource "vault_mount" "q_platform_secrets" {
  path = "secret"
  type = "kv"
  options = {
    version = "2"
  }
  depends_on = [helm_release.vault]
}

# --- Service-Specific Vault Roles ---

module "managerq_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "manager-q"
  namespace    = var.namespace
  secrets_path = "secret/data/manager-q/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "agentq_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "agentq"
  namespace    = var.namespace
  secrets_path = "secret/data/agentq/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "h2m_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "h2m-service"
  namespace    = var.namespace
  secrets_path = "secret/data/h2m-service/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "quantumpulse_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "quantumpulse"
  namespace    = var.namespace
  secrets_path = "secret/data/quantumpulse/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "vectorstoreq_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "vectorstore-q"
  namespace    = var.namespace
  secrets_path = "secret/data/vectorstore-q/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "knowledgegraphq_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "knowledgegraphq"
  namespace    = var.namespace
  secrets_path = "secret/data/knowledgegraphq/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "integrationhub_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "integrationhub"
  namespace    = var.namespace
  secrets_path = "secret/data/integrationhub/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "userprofileq_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "userprofileq"
  namespace    = var.namespace
  secrets_path = "secret/data/userprofileq/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "authq_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "authq"
  namespace    = var.namespace
  secrets_path = "secret/data/authq/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "webapp_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "webapp-q"
  namespace    = var.namespace
  secrets_path = "secret/data/webapp-q/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "workflowworker_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "workflowworker"
  namespace    = var.namespace
  secrets_path = "secret/data/workflowworker/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "aiops_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "aiops"
  namespace    = var.namespace
  secrets_path = "secret/data/aiops/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

module "agentsandbox_vault_role" {
  source       = "./modules/vault-service-role"
  service_name = "agentsandbox"
  namespace    = var.namespace
  secrets_path = "secret/data/agentsandbox/*"
  depends_on   = [vault_mount.q_platform_secrets]
}

# --- Network Policies ---

resource "kubernetes_network_policy" "q_platform_network_policy" {
  metadata {
    name      = "q-platform-network-policy"
    namespace = var.namespace
  }

  spec {
    pod_selector {}

    policy_types = ["Ingress", "Egress"]

    # Allow ingress from Istio gateway
    ingress {
      from {
        namespace_selector {
          match_labels = {
            name = "istio-system"
          }
        }
      }
    }

    # Allow ingress within namespace
    ingress {
      from {
        namespace_selector {
          match_labels = {
            name = var.namespace
          }
        }
      }
    }

    # Allow egress to system services
    egress {
      to {
        namespace_selector {
          match_labels = {
            name = "kube-system"
          }
        }
      }
    }

    # Allow egress to Vault
    egress {
      to {
        namespace_selector {
          match_labels = {
            name = "vault"
          }
        }
      }
    }

    # Allow egress to Istio system
    egress {
      to {
        namespace_selector {
          match_labels = {
            name = "istio-system"
          }
        }
      }
    }

    # Allow egress within namespace
    egress {
      to {
        namespace_selector {
          match_labels = {
            name = var.namespace
          }
        }
      }
    }

    # Allow DNS resolution
    egress {
      to {
        namespace_selector {
          match_labels = {
            name = "kube-system"
          }
        }
      }
      ports {
        port     = "53"
        protocol = "UDP"
      }
    }
  }
  depends_on = [helm_release.istiod]
}

# --- TLS Certificates ---

resource "kubernetes_manifest" "q_platform_certificate" {
  count = var.enable_tls ? 1 : 0
  
  manifest = {
    "apiVersion" = "cert-manager.io/v1"
    "kind"       = "Certificate"
    "metadata" = {
      "name"      = "q-platform-tls"
      "namespace" = "istio-system"
    }
    "spec" = {
      "secretName" = "q-platform-tls"
      "issuerRef" = {
        "name" = "letsencrypt-prod"
        "kind" = "ClusterIssuer"
      }
      "dnsNames" = [
        var.domain_name,
        "*.${var.domain_name}"
      ]
    }
  }
}

# --- Service Accounts ---

resource "kubernetes_service_account" "q_platform_services" {
  for_each = toset([
    "manager-q",
    "agentq",
    "h2m-service",
    "quantumpulse",
    "vectorstore-q",
    "knowledgegraphq",
    "integrationhub",
    "userprofileq",
    "authq",
    "webapp-q",
    "workflowworker",
    "aiops",
    "agentsandbox"
  ])

  metadata {
    name      = each.value
    namespace = var.namespace
    annotations = {
      "vault.hashicorp.com/agent-inject"               = "true"
      "vault.hashicorp.com/role"                       = "${each.value}-role"
      "vault.hashicorp.com/agent-inject-secret-config" = "secret/data/${each.value}"
    }
  }
} 