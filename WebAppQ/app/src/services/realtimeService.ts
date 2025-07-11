import { EventEmitter } from 'events';

export interface MarketDataEvent {
    symbol: string;
    price: number;
    change: number;
    change_percent: number;
    volume: number;
    timestamp: number;
    source: string;
    anomalies?: Array<{
        type: string;
        severity: string;
        change_percent?: number;
        direction?: string;
        volume?: number;
    }>;
}

export interface SpikeEvent {
    time: number;
    intensity: number;
    neuron_id: string;
    layer: string;
}

export interface AnomalyEvent {
    id: string;
    type: string;
    severity: string;
    description: string;
    timestamp: number;
    affected_services: string[];
    confidence: number;
}

export interface EthicalVetoEvent {
    workflow_id: string;
    reason: string;
    principle_violated: string;
    timestamp: string;
    severity: 'warning' | 'critical';
    recommendation: string;
}

export interface QGANTrainingEvent {
    status: 'started' | 'training' | 'completed' | 'failed';
    progress?: number;
    current_epoch?: number;
    total_epochs?: number;
    g_loss?: number;
    d_loss?: number;
    generated_samples?: any[];
    error?: string;
}

class RealtimeDataService extends EventEmitter {
    private ws: WebSocket | null = null;
    private reconnectInterval: number = 5000;
    private reconnectTimer: NodeJS.Timeout | null = null;
    private isConnecting: boolean = false;
    private eventBuffer: any[] = [];
    private maxBufferSize: number = 100;

    constructor() {
        super();
        this.setMaxListeners(50); // Increase for multiple widgets
    }

    connect(token: string) {
        if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
            return;
        }

        this.isConnecting = true;
        const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/realtime`;
        
        try {
            this.ws = new WebSocket(wsUrl, [], {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            this.ws.onopen = () => {
                console.log('Realtime WebSocket connected');
                this.isConnecting = false;
                this.emit('connected');
                
                // Subscribe to data streams
                this.subscribe(['market-data', 'neuromorphic-spikes', 'anomalies', 'ethical-reviews', 'qgan-training']);
                
                // Flush buffered events
                this.flushEventBuffer();
            };

            this.ws.onclose = (event) => {
                console.log('Realtime WebSocket disconnected', event);
                this.isConnecting = false;
                this.emit('disconnected');
                this.scheduleReconnect(token);
            };

            this.ws.onerror = (error) => {
                console.error('Realtime WebSocket error:', error);
                this.isConnecting = false;
                this.emit('error', error);
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.isConnecting = false;
            this.scheduleReconnect(token);
        }
    }

    private subscribe(topics: string[]) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'subscribe',
                topics: topics
            }));
        }
    }

    private handleMessage(data: any) {
        switch (data.type) {
            case 'market-data':
                this.emit('marketData', data.payload as MarketDataEvent);
                break;
            
            case 'spike-train':
                this.emit('spikeTrain', data.payload as SpikeEvent);
                break;
            
            case 'anomaly-detected':
                this.emit('anomaly', data.payload as AnomalyEvent);
                break;
            
            case 'ethical-veto':
                this.emit('ethicalVeto', data.payload as EthicalVetoEvent);
                break;
            
            case 'qgan-update':
                this.emit('qganUpdate', data.payload as QGANTrainingEvent);
                break;
            
            case 'neuromorphic-metrics':
                this.emit('neuromorphicMetrics', data.payload);
                break;
            
            default:
                // Generic event emission for extensibility
                this.emit(data.type, data.payload);
        }
    }

    private scheduleReconnect(token: string) {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }

        this.reconnectTimer = setTimeout(() => {
            console.log('Attempting to reconnect WebSocket...');
            this.connect(token);
        }, this.reconnectInterval);
    }

    private flushEventBuffer() {
        while (this.eventBuffer.length > 0 && this.ws && this.ws.readyState === WebSocket.OPEN) {
            const event = this.eventBuffer.shift();
            this.ws.send(JSON.stringify(event));
        }
    }

    disconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    // Market Data Stream
    subscribeToMarketData(callback: (data: MarketDataEvent) => void) {
        this.on('marketData', callback);
        return () => this.off('marketData', callback);
    }

    // Neuromorphic Data Stream
    subscribeToSpikeTrains(callback: (spike: SpikeEvent) => void) {
        this.on('spikeTrain', callback);
        return () => this.off('spikeTrain', callback);
    }

    subscribeToAnomalies(callback: (anomaly: AnomalyEvent) => void) {
        this.on('anomaly', callback);
        return () => this.off('anomaly', callback);
    }

    subscribeToNeuromorphicMetrics(callback: (metrics: any) => void) {
        this.on('neuromorphicMetrics', callback);
        return () => this.off('neuromorphicMetrics', callback);
    }

    // Ethical Review Stream
    subscribeToEthicalVetos(callback: (veto: EthicalVetoEvent) => void) {
        this.on('ethicalVeto', callback);
        return () => this.off('ethicalVeto', callback);
    }

    // QGAN Training Stream
    subscribeToQGANUpdates(callback: (update: QGANTrainingEvent) => void) {
        this.on('qganUpdate', callback);
        return () => this.off('qganUpdate', callback);
    }

    // Request specific data
    async requestQGANTraining(workflowId: string): Promise<void> {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'request',
                action: 'start_qgan_training',
                workflow_id: workflowId
            }));
        } else {
            // Buffer the request
            this.eventBuffer.push({
                type: 'request',
                action: 'start_qgan_training',
                workflow_id: workflowId
            });
            
            if (this.eventBuffer.length > this.maxBufferSize) {
                this.eventBuffer.shift(); // Remove oldest
            }
        }
    }
}

// Singleton instance
export const realtimeService = new RealtimeDataService();

// Hook for React components
import { useEffect, useState, useCallback } from 'react';

export function useRealtimeData<T>(
    subscribeFn: (callback: (data: T) => void) => () => void,
    initialValue: T
): T {
    const [data, setData] = useState<T>(initialValue);

    useEffect(() => {
        const unsubscribe = subscribeFn(setData);
        return unsubscribe;
    }, [subscribeFn]);

    return data;
}

export function useMarketData() {
    const [marketData, setMarketData] = useState<MarketDataEvent[]>([]);
    const maxDataPoints = 100;

    useEffect(() => {
        const unsubscribe = realtimeService.subscribeToMarketData((data) => {
            setMarketData(prev => {
                const newData = [...prev, data];
                return newData.slice(-maxDataPoints);
            });
        });

        return unsubscribe;
    }, []);

    return marketData;
}

export function useSpikeTrains() {
    const [spikes, setSpikes] = useState<SpikeEvent[]>([]);
    const maxSpikes = 200;

    useEffect(() => {
        const unsubscribe = realtimeService.subscribeToSpikeTrains((spike) => {
            setSpikes(prev => {
                const newSpikes = [...prev, spike];
                return newSpikes.slice(-maxSpikes);
            });
        });

        return unsubscribe;
    }, []);

    return spikes;
}

export function useAnomalies() {
    const [anomalies, setAnomalies] = useState<AnomalyEvent[]>([]);

    useEffect(() => {
        const unsubscribe = realtimeService.subscribeToAnomalies((anomaly) => {
            setAnomalies(prev => [anomaly, ...prev].slice(0, 50));
        });

        return unsubscribe;
    }, []);

    return anomalies;
}

export function useEthicalVetos() {
    const [vetos, setVetos] = useState<EthicalVetoEvent[]>([]);

    useEffect(() => {
        const unsubscribe = realtimeService.subscribeToEthicalVetos((veto) => {
            setVetos(prev => [veto, ...prev].slice(0, 20));
        });

        return unsubscribe;
    }, []);

    return vetos;
}

export function useQGANTraining() {
    const [status, setStatus] = useState<QGANTrainingEvent>({ status: 'completed' });

    const startTraining = useCallback(async (workflowId: string) => {
        setStatus({ status: 'started' });
        await realtimeService.requestQGANTraining(workflowId);
    }, []);

    useEffect(() => {
        const unsubscribe = realtimeService.subscribeToQGANUpdates(setStatus);
        return unsubscribe;
    }, []);

    return { status, startTraining };
} 