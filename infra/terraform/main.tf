terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.0.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.0.0"
    }
    vault = {
      source  = "hashicorp/vault"
      version = ">= 3.0.0"
    }
  }
}

provider "kubernetes" {
  config_path = var.kubeconfig_path
}

provider "helm" {
  kubernetes {
    config_path = var.kubeconfig_path
  }
}

provider "vault" {
  # Configure vault provider
  address = var.vault_address
}

# --- Helm Chart Repositories ---

resource "helm_repository" "bitnami" {
  name = "bitnami"
  url  = "https://charts.bitnami.com/bitnami"
}

resource "helm_repository" "milvus" {
  name = "milvus"
  url  = "https://milvus-io.github.io/milvus-helm/"
}

resource "helm_repository" "goharbor" {
  name = "goharbor"
  url  = "https://helm.goharbor.io"
}

resource "helm_repository" "prometheus_community" {
  name = "prometheus-community"
  url  = "https://prometheus-community.github.io/helm-charts"
}

resource "helm_repository" "grafana" {
  name = "grafana"
  url  = "https://grafana.github.io/helm-charts"
}

resource "helm_repository" "argo" {
  name = "argo"
  url  = "https://argoproj.github.io/argo-helm"
}

resource "helm_repository" "hashicorp" {
  name = "hashicorp"
  url  = "https://helm.releases.hashicorp.com"
}

resource "helm_repository" "elastic" {
  name = "elastic"
  url  = "https://helm.elastic.co"
}

resource "helm_repository" "apache" {
  name = "apache"
  url  = "https://pulsar.apache.org/charts"
}

resource "helm_repository" "janusgraph" {
  name = "janusgraph"
  url  = "https://ibm.github.io/janusgraph-utils"
}

resource "helm_repository" "apache_flink" {
  name = "apache-flink"
  url  = "https://apache.github.io/flink-kubernetes-operator"
}

resource "helm_repository" "istio" {
  name = "istio"
  url  = "https://istio-release.storage.googleapis.com/charts"
}

resource "helm_repository" "spark" {
  name = "spark"
  url  = "https://googlecloudplatform.github.io/spark-on-k8s-operator"
}

# --- Core Infrastructure Services ---

resource "helm_release" "vault" {
  name       = "vault"
  repository = helm_repository.hashicorp.name
  chart      = "vault"
  version    = "0.27.0"
  namespace  = "vault"
  create_namespace = true

  values = [
    file("${path.module}/values/vault.yaml")
  ]
}

resource "helm_release" "keycloak" {
  name       = "keycloak"
  repository = helm_repository.bitnami.name
  chart      = "keycloak"
  version    = "18.2.1"
  namespace  = var.namespace
  create_namespace = true

  values = [
    file("${path.module}/values/keycloak.yaml")
  ]
  depends_on = [helm_release.vault]
}

# --- Data Layer Services ---

resource "helm_release" "pulsar" {
  name       = "pulsar"
  repository = helm_repository.apache.name
  chart      = "pulsar"
  version    = var.pulsar_chart_version
  namespace  = var.namespace

  values = [
    file("${path.module}/values/pulsar.yaml")
  ]
  depends_on = [helm_release.vault]
}

resource "helm_release" "cassandra" {
  name       = "cassandra"
  repository = helm_repository.bitnami.name
  chart      = "cassandra"
  version    = var.cassandra_chart_version
  namespace  = var.namespace

  values = [
    file("${path.module}/values/cassandra.yaml")
  ]
  depends_on = [helm_release.vault]
}

resource "helm_release" "elasticsearch" {
  name       = "elasticsearch"
  repository = helm_repository.elastic.name
  chart      = "elasticsearch"
  version    = var.elasticsearch_chart_version
  namespace  = var.namespace

  values = [
    file("${path.module}/values/elasticsearch.yaml")
  ]
  depends_on = [helm_release.vault]
}

resource "helm_release" "milvus" {
  name       = "milvus"
  repository = helm_repository.milvus.name
  chart      = "milvus"
  version    = "4.0.12"
  namespace  = var.namespace

  values = [
    file("${path.module}/values/milvus.yaml")
  ]
  depends_on = [helm_release.vault]
}

resource "helm_release" "janusgraph" {
  name       = "janusgraph"
  repository = helm_repository.janusgraph.name
  chart      = "janusgraph"
  version    = var.janusgraph_chart_version
  namespace  = var.namespace

  values = [
    file("${path.module}/values/janusgraph.yaml")
  ]
  depends_on = [helm_release.cassandra]
}

resource "helm_release" "ignite" {
  name       = "ignite"
  repository = helm_repository.bitnami.name
  chart      = "ignite"
  version    = var.ignite_chart_version
  namespace  = var.namespace

  values = [
    file("${path.module}/values/ignite.yaml")
  ]
  depends_on = [helm_release.vault]
}

resource "helm_release" "minio" {
  name       = "minio"
  repository = helm_repository.bitnami.name
  chart      = "minio"
  version    = var.minio_chart_version
  namespace  = var.namespace

  values = [
    file("${path.module}/values/minio.yaml")
  ]
  depends_on = [helm_release.vault]
}

# --- Processing Layer Services ---

resource "helm_release" "flink" {
  name       = "flink"
  repository = helm_repository.apache_flink.name
  chart      = "flink-kubernetes-operator"
  version    = var.flink_chart_version
  namespace  = var.namespace

  values = [
    file("${path.module}/values/flink.yaml")
  ]
  depends_on = [helm_release.pulsar]
}

resource "helm_release" "spark" {
  name       = "spark"
  repository = helm_repository.spark.name
  chart      = "spark-operator"
  version    = "1.1.27"
  namespace  = var.namespace

  values = [
    templatefile("${path.module}/values/spark.yaml", {
      namespace = var.namespace
    })
  ]
  depends_on = [helm_release.vault]
}

resource "helm_release" "seatunnel" {
  name       = "seatunnel"
  repository = helm_repository.apache.name
  chart      = "seatunnel"
  version    = var.seatunnel_chart_version
  namespace  = var.namespace

  values = [
    file("${path.module}/values/seatunnel.yaml")
  ]
  depends_on = [helm_release.flink]
}

# --- Service Mesh ---

resource "helm_release" "istio_base" {
  name       = "istio-base"
  repository = helm_repository.istio.name
  chart      = "base"
  version    = var.istio_base_chart_version
  namespace  = "istio-system"
  create_namespace = true

  set {
    name  = "defaultRevision"
    value = "default"
  }
}

resource "helm_release" "istiod" {
  name       = "istiod"
  repository = helm_repository.istio.name
  chart      = "istiod"
  version    = var.istiod_chart_version
  namespace  = "istio-system"

  values = [
    file("${path.module}/values/istiod.yaml")
  ]
  depends_on = [helm_release.istio_base]
}

resource "helm_release" "istio_gateway" {
  name       = "istio-gateway"
  repository = helm_repository.istio.name
  chart      = "gateway"
  version    = var.istio_gateway_chart_version
  namespace  = "istio-system"

  values = [
    templatefile("${path.module}/values/istio-gateway.yaml", {
      namespace = var.namespace
    })
  ]
  depends_on = [helm_release.istiod]
}

# --- Container Registry ---

resource "helm_release" "harbor" {
  name       = "harbor"
  repository = helm_repository.goharbor.name
  chart      = "harbor"
  version    = var.harbor_chart_version
  namespace  = var.namespace

  values = [
    file("${path.module}/values/harbor.yaml")
  ]
  depends_on = [helm_release.vault, helm_release.minio]
}

# --- Observability Stack ---

resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = helm_repository.prometheus_community.name
  chart      = "prometheus"
  version    = "25.4.0"
  namespace  = var.namespace

  values = [
    file("${path.module}/values/prometheus.yaml")
  ]
  depends_on = [helm_release.vault]
}

resource "helm_release" "grafana" {
  name       = "grafana"
  repository = helm_repository.grafana.name
  chart      = "grafana"
  version    = "7.3.5"
  namespace  = var.namespace

  values = [
    file("${path.module}/values/grafana.yaml")
  ]
  depends_on = [helm_release.prometheus]
}

resource "helm_release" "tempo" {
  name       = "tempo"
  repository = helm_repository.grafana.name
  chart      = "tempo"
  version    = var.tempo_chart_version
  namespace  = var.namespace

  values = [
    file("${path.module}/values/tempo.yaml")
  ]
  depends_on = [helm_release.minio]
}

# --- GitOps ---

resource "helm_release" "argocd" {
  name       = "argo-cd"
  repository = helm_repository.argo.name
  chart      = "argo-cd"
  version    = var.argocd_chart_version
  namespace  = "argocd"
  create_namespace = true

  values = [
    file("${path.module}/values/argocd.yaml")
  ]
  depends_on = [helm_release.vault]
}

# --- Outputs ---

output "vault_endpoint" {
  value = "http://vault.vault.svc.cluster.local:8200"
}

output "keycloak_endpoint" {
  value = "http://keycloak.${var.namespace}.svc.cluster.local:8080"
}

output "pulsar_endpoint" {
  value = "pulsar://pulsar-broker.${var.namespace}.svc.cluster.local:6650"
}

output "milvus_endpoint" {
  value = "milvus.${var.namespace}.svc.cluster.local:19530"
}

output "elasticsearch_endpoint" {
  value = "http://elasticsearch.${var.namespace}.svc.cluster.local:9200"
}

output "harbor_endpoint" {
  value = "https://harbor.${var.namespace}.svc.cluster.local"
}

output "prometheus_endpoint" {
  value = "http://prometheus.${var.namespace}.svc.cluster.local:9090"
}

output "grafana_endpoint" {
  value = "http://grafana.${var.namespace}.svc.cluster.local:3000"
} 