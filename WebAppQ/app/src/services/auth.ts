import keycloak from "../keycloak";

export const authenticatedRequest = async (url: string, options: RequestInit = {}) => {
    if (!keycloak.authenticated) {
        throw new Error("User not authenticated");
    }

    const headers = {
        ...options.headers,
        Authorization: `Bearer ${keycloak.token}`,
    };

    return fetch(url, { ...options, headers });
}; 