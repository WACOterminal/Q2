variable "kubeconfig_path" {
  description = "Path to the kubeconfig file used by Terraform to access the Kubernetes cluster."
  type        = string
  default     = "~/.kube/config"
}

variable "namespace" {
  description = "Kubernetes namespace where all platform services will be deployed."
  type        = string
  default     = "q-platform"
}

variable "vault_address" {
  description = "Address of the Vault server"
  type        = string
  default     = "http://vault.vault.svc.cluster.local:8200"
}

variable "pulsar_chart_version" {
  description = "Helm chart version for Apache Pulsar."
  type        = string
  default     = "5.1.4"
}

variable "cassandra_chart_version" {
  description = "Helm chart version for Cassandra (Bitnami)."
  type        = string
  default     = "12.6.3"
}

variable "elasticsearch_chart_version" {
  description = "Helm chart version for Elasticsearch."
  type        = string
  default     = "8.9.2"
}

variable "janusgraph_chart_version" {
  description = "Helm chart version for JanusGraph."
  type        = string
  default     = "0.6.1"
}

variable "ignite_chart_version" {
  description = "Helm chart version for Apache Ignite."
  type        = string
  default     = "1.1.0"
}

variable "flink_chart_version" {
  description = "Helm chart version for Apache Flink."
  type        = string
  default     = "1.7.0"
}

variable "spark_chart_version" {
  description = "Helm chart version for Apache Spark."
  type        = string
  default     = "1.1.27"
}

variable "minio_chart_version" {
  description = "Helm chart version for MinIO."
  type        = string
  default     = "5.0.10"
}

variable "argocd_chart_version" {
  description = "Helm chart version for ArgoCD."
  type        = string
  default     = "5.51.1"
}

variable "harbor_chart_version" {
  description = "Helm chart version for Harbor."
  type        = string
  default     = "1.13.2"
}

variable "tempo_chart_version" {
  description = "Helm chart version for Grafana Tempo."
  type        = string
  default     = "1.8.0"
}

variable "istio_base_chart_version" {
  description = "Helm chart version for Istio Base."
  type        = string
  default     = "1.21.0"
}

variable "istiod_chart_version" {
  description = "Helm chart version for IstioD (Control Plane)."
  type        = string
  default     = "1.21.0"
}

variable "istio_gateway_chart_version" {
  description = "Helm chart version for Istio Ingress Gateway."
  type        = string
  default     = "1.21.0"
}

variable "seatunnel_chart_version" {
  description = "The version of the Apache SeaTunnel Helm chart to deploy."
  type        = string
  default     = "2.3.10"
}

variable "keycloak_issuer_url" {
  type        = string
  description = "The OIDC issuer URL for the Keycloak realm."
  default     = "http://keycloak.q-platform.svc.cluster.local:8080/realms/q-platform"
}

variable "primary_region" {
  description = "The primary region for the deployment."
  type        = string
  default     = "us-east-1"
}

variable "secondary_region" {
  description = "The secondary region for disaster recovery."
  type        = string
  default     = "us-west-2"
}

variable "domain_name" {
  description = "The domain name for the platform"
  type        = string
  default     = "q-platform.local"
}

variable "enable_tls" {
  description = "Enable TLS for the platform"
  type        = bool
  default     = true
}

variable "storage_class" {
  description = "Storage class for persistent volumes"
  type        = string
  default     = "standard"
} 