import React, { useEffect, useState } from "react";
import { getConnectors, Connector } from "../services/integrationHub";

const Integrations: React.FC = () => {
    const [connectors, setConnectors] = useState<Connector[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchConnectors = async () => {
            try {
                const data = await getConnectors();
                setConnectors(data);
            } catch (err) {
                setError("Failed to fetch connectors.");
            } finally {
                setLoading(false);
            }
        };

        fetchConnectors();
    }, []);

    if (loading) return <div>Loading...</div>;
    if (error) return <div>{error}</div>;

    return (
        <div>
            <h1>Integrations</h1>
            <ul>
                {connectors.map((connector) => (
                    <li key={connector.name}>
                        <h2>{connector.name}</h2>
                        <p>{connector.description}</p>
                        <button disabled>{connector.enabled ? "Enabled" : "Disabled"}</button>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default Integrations; 