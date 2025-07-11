{{/*
Expand the name of the chart.
*/}}
{{- define "q-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "q-platform.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "q-platform.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "q-platform.labels" -}}
helm.sh/chart: {{ include "q-platform.chart" . }}
{{ include "q-platform.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: q-platform
{{- end }}

{{/*
Selector labels
*/}}
{{- define "q-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "q-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "q-platform.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "q-platform.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create a default fully qualified postgresql name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.postgresql.fullname" -}}
{{- include "q-platform.fullname" . }}-postgresql
{{- end }}

{{/*
Create a default fully qualified pulsar name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.pulsar.fullname" -}}
{{- include "q-platform.fullname" . }}-pulsar
{{- end }}

{{/*
Create a default fully qualified cassandra name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.cassandra.fullname" -}}
{{- include "q-platform.fullname" . }}-cassandra
{{- end }}

{{/*
Create a default fully qualified milvus name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.milvus.fullname" -}}
{{- include "q-platform.fullname" . }}-milvus
{{- end }}

{{/*
Create a default fully qualified ignite name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.ignite.fullname" -}}
{{- include "q-platform.fullname" . }}-ignite
{{- end }}

{{/*
Create a default fully qualified elasticsearch name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.elasticsearch.fullname" -}}
{{- include "q-platform.fullname" . }}-elasticsearch
{{- end }}

{{/*
Create a default fully qualified minio name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.minio.fullname" -}}
{{- include "q-platform.fullname" . }}-minio
{{- end }}

{{/*
Create a default fully qualified keycloak name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.keycloak.fullname" -}}
{{- include "q-platform.fullname" . }}-keycloak
{{- end }}

{{/*
Create a default fully qualified vault name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.vault.fullname" -}}
{{- include "q-platform.fullname" . }}-vault
{{- end }}

{{/*
Create a default fully qualified prometheus name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.prometheus.fullname" -}}
{{- include "q-platform.fullname" . }}-prometheus
{{- end }}

{{/*
Create a default fully qualified grafana name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.grafana.fullname" -}}
{{- include "q-platform.fullname" . }}-grafana
{{- end }}

{{/*
Create a default fully qualified jaeger name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.jaeger.fullname" -}}
{{- include "q-platform.fullname" . }}-jaeger
{{- end }}

{{/*
Create a default fully qualified argocd name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.argocd.fullname" -}}
{{- include "q-platform.fullname" . }}-argocd
{{- end }}

{{/*
Create a default fully qualified harbor name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "q-platform.harbor.fullname" -}}
{{- include "q-platform.fullname" . }}-harbor
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "q-platform.image" -}}
{{- if .registry -}}
{{- printf "%s/%s:%s" .registry .repository .tag -}}
{{- else -}}
{{- printf "%s:%s" .repository .tag -}}
{{- end -}}
{{- end -}}

{{/*
Return the proper Docker Image Registry Secret Names
*/}}
{{- define "q-platform.imagePullSecrets" -}}
{{- include "q-platform.images.pullSecrets" (dict "images" (list .Values.image) "global" .Values.global) -}}
{{- end -}}

{{/*
Return the proper Docker Image Registry Secret Names (deprecated: use q-platform.imagePullSecrets instead)
*/}}
{{- define "q-platform.images.pullSecrets" -}}
{{- $pullSecrets := list }}

{{- if .global }}
  {{- range .global.imagePullSecrets -}}
    {{- $pullSecrets = append $pullSecrets . -}}
  {{- end -}}
{{- end -}}

{{- range .images -}}
  {{- range .pullSecrets -}}
    {{- $pullSecrets = append $pullSecrets . -}}
  {{- end -}}
{{- end -}}

{{- if (not (empty $pullSecrets)) }}
imagePullSecrets:
{{- range $pullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end -}}

{{/*
Renders a value that contains template.
Usage:
{{ include "q-platform.tplvalues.render" ( dict "value" .Values.path.to.the.Value "context" $) }}
*/}}
{{- define "q-platform.tplvalues.render" -}}
    {{- if typeIs "string" .value }}
        {{- tpl .value .context }}
    {{- else }}
        {{- tpl (.value | toYaml) .context }}
    {{- end }}
{{- end -}}

{{/*
Return the target Kubernetes version
*/}}
{{- define "q-platform.capabilities.kubeVersion" -}}
{{- if .Values.global }}
    {{- if .Values.global.kubeVersion }}
    {{- .Values.global.kubeVersion -}}
    {{- else }}
    {{- default .Capabilities.KubeVersion.Version .Values.kubeVersion -}}
    {{- end -}}
{{- else }}
{{- default .Capabilities.KubeVersion.Version .Values.kubeVersion -}}
{{- end -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for policy.
*/}}
{{- define "q-platform.capabilities.policy.apiVersion" -}}
{{- if semverCompare "<1.21-0" (include "q-platform.capabilities.kubeVersion" .) -}}
{{- print "policy/v1beta1" -}}
{{- else -}}
{{- print "policy/v1" -}}
{{- end -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for networkpolicy.
*/}}
{{- define "q-platform.capabilities.networkPolicy.apiVersion" -}}
{{- if semverCompare "<1.7-0" (include "q-platform.capabilities.kubeVersion" .) -}}
{{- print "extensions/v1beta1" -}}
{{- else -}}
{{- print "networking.k8s.io/v1" -}}
{{- end -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for ingress.
*/}}
{{- define "q-platform.capabilities.ingress.apiVersion" -}}
{{- if semverCompare "<1.14-0" (include "q-platform.capabilities.kubeVersion" .) -}}
{{- print "extensions/v1beta1" -}}
{{- else if semverCompare "<1.19-0" (include "q-platform.capabilities.kubeVersion" .) -}}
{{- print "networking.k8s.io/v1beta1" -}}
{{- else -}}
{{- print "networking.k8s.io/v1" -}}
{{- end -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for rbac.
*/}}
{{- define "q-platform.capabilities.rbac.apiVersion" -}}
{{- if semverCompare "<1.17-0" (include "q-platform.capabilities.kubeVersion" .) -}}
{{- print "rbac.authorization.k8s.io/v1beta1" -}}
{{- else -}}
{{- print "rbac.authorization.k8s.io/v1" -}}
{{- end -}}
{{- end -}}

{{/*
Create the name of the service account to use for Manager Q
*/}}
{{- define "q-platform.manager-q.serviceAccountName" -}}
{{- if .Values.services.managerQ.serviceAccount.create }}
{{- default (printf "%s-manager-q" (include "q-platform.fullname" .)) .Values.services.managerQ.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.services.managerQ.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for Agent Q
*/}}
{{- define "q-platform.agent-q.serviceAccountName" -}}
{{- if .Values.services.agentQ.serviceAccount.create }}
{{- default (printf "%s-agent-q" (include "q-platform.fullname" .)) .Values.services.agentQ.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.services.agentQ.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for H2M Service
*/}}
{{- define "q-platform.h2m-service.serviceAccountName" -}}
{{- if .Values.services.h2mService.serviceAccount.create }}
{{- default (printf "%s-h2m-service" (include "q-platform.fullname" .)) .Values.services.h2mService.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.services.h2mService.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for Quantum Pulse
*/}}
{{- define "q-platform.quantum-pulse.serviceAccountName" -}}
{{- if .Values.services.quantumPulse.serviceAccount.create }}
{{- default (printf "%s-quantum-pulse" (include "q-platform.fullname" .)) .Values.services.quantumPulse.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.services.quantumPulse.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for Vector Store Q
*/}}
{{- define "q-platform.vector-store-q.serviceAccountName" -}}
{{- if .Values.services.vectorStoreQ.serviceAccount.create }}
{{- default (printf "%s-vector-store-q" (include "q-platform.fullname" .)) .Values.services.vectorStoreQ.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.services.vectorStoreQ.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for Knowledge Graph Q
*/}}
{{- define "q-platform.knowledge-graph-q.serviceAccountName" -}}
{{- if .Values.services.knowledgeGraphQ.serviceAccount.create }}
{{- default (printf "%s-knowledge-graph-q" (include "q-platform.fullname" .)) .Values.services.knowledgeGraphQ.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.services.knowledgeGraphQ.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for Integration Hub
*/}}
{{- define "q-platform.integration-hub.serviceAccountName" -}}
{{- if .Values.services.integrationHub.serviceAccount.create }}
{{- default (printf "%s-integration-hub" (include "q-platform.fullname" .)) .Values.services.integrationHub.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.services.integrationHub.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for User Profile Q
*/}}
{{- define "q-platform.user-profile-q.serviceAccountName" -}}
{{- if .Values.services.userProfileQ.serviceAccount.create }}
{{- default (printf "%s-user-profile-q" (include "q-platform.fullname" .)) .Values.services.userProfileQ.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.services.userProfileQ.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for Web App Q
*/}}
{{- define "q-platform.web-app-q.serviceAccountName" -}}
{{- if .Values.services.webAppQ.serviceAccount.create }}
{{- default (printf "%s-web-app-q" (include "q-platform.fullname" .)) .Values.services.webAppQ.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.services.webAppQ.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get the password secret.
*/}}
{{- define "q-platform.secretName" -}}
{{- if .Values.auth.existingSecret -}}
{{- .Values.auth.existingSecret -}}
{{- else -}}
{{- include "q-platform.fullname" . -}}-auth
{{- end -}}
{{- end -}}

{{/*
Get the password key.
*/}}
{{- define "q-platform.secretPasswordKey" -}}
{{- if .Values.auth.existingSecret -}}
{{- .Values.auth.existingSecretPasswordKey | default "password" -}}
{{- else -}}
{{- "password" -}}
{{- end -}}
{{- end -}}

{{/*
Return true if cert-manager required annotations for TLS signed certificates are set in the Ingress annotations
Ref: https://cert-manager.io/docs/usage/ingress/#supported-annotations
*/}}
{{- define "q-platform.ingress.certManagerRequest" -}}
{{ if or (hasKey . "cert-manager.io/cluster-issuer") (hasKey . "cert-manager.io/issuer") }}
    {{- true -}}
{{ end }}
{{- end -}}

{{/*
Compile all warnings into a single message.
*/}}
{{- define "q-platform.validateValues" -}}
{{- $messages := list -}}
{{- $messages := append $messages (include "q-platform.validateValues.database" .) -}}
{{- $messages := append $messages (include "q-platform.validateValues.ingress" .) -}}
{{- $messages := without $messages "" -}}
{{- $message := join "\n" $messages -}}

{{- if $message -}}
{{-   printf "\nVALUES VALIDATION:\n%s" $message -}}
{{- end -}}
{{- end -}}

{{/*
Validate values of Q Platform - Database
*/}}
{{- define "q-platform.validateValues.database" -}}
{{- if and (not .Values.infrastructure.cassandra.enabled) (not .Values.infrastructure.postgresql.enabled) -}}
q-platform: database
    You must enable at least one database.
    Please enable either Cassandra or PostgreSQL.
{{- end -}}
{{- end -}}

{{/*
Validate values of Q Platform - Ingress
*/}}
{{- define "q-platform.validateValues.ingress" -}}
{{- if and .Values.ingress.enabled (not .Values.ingress.hostname) -}}
q-platform: ingress
    You must provide a hostname when enabling ingress.
    Please set the 'ingress.hostname' value.
{{- end -}}
{{- end -}} 