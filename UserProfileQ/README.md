# UserProfileQ Service

The `UserProfileQ` service is responsible for managing user profiles, preferences, and other user-specific data. It provides a RESTful API for creating, retrieving, and updating user profiles.

## 🏛️ Architecture

This service is a Python-based FastAPI application that uses Apache Cassandra as its primary data store. It integrates with the platform's central authentication service (`AuthQ`) to secure its endpoints and associate profiles with authenticated users.

## 🚀 API Endpoints

All endpoints are available under the `/api/v1/profiles` prefix.

| Method | Endpoint      | Description                               |
|--------|---------------|-------------------------------------------|
| `POST` | `/`           | Creates a new profile for the authenticated user. |
| `GET`   | `/me`         | Retrieves the profile of the authenticated user. |
| `PUT`   | `/me`         | Updates the profile of the authenticated user. |
| `GET`   | `/{user_id}`  | Retrieves a user's profile by their ID (admin only). |

### Data Model

The user profile is defined by the Avro schema in `UserProfileQ/schemas/user_profile.avsc`. The key fields are:
- `user_id` (string, UUID)
- `username` (string)
- `email` (string)
- `display_name` (string, optional)
- `preferences` (map of string-to-string, optional)

## 📦 Running Locally

1.  **Start Dependencies**: Ensure you have a running instance of Apache Cassandra and that the `AuthQ` service is operational.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Service**:
    ```bash
    uvicorn UserProfileQ.app.main:app --host 0.0.0.0 --port 8000
    ```

## 🚢 Deployment

This service is deployed automatically via the GitOps workflow. The Kubernetes manifests are located in `infra/kubernetes/base/userprofileq/` and are managed by Kustomize and ArgoCD. The CI/CD pipeline is defined in `.github/workflows/userprofileq-ci.yml`. 