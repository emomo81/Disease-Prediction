import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Symptom {
  value: string;
  label: string;
}

export interface Prediction {
  disease: string;
  confidence: number;
  description: string;
}

export interface PredictionResponse {
  predictions: Prediction[];
  selected_symptoms: string[];
  total_predictions: number;
}

export interface SymptomsResponse {
  symptoms: Symptom[];
  total: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  total_symptoms: number;
  total_diseases: number;
}

export const api = {
  // Health check
  async checkHealth(): Promise<HealthResponse> {
    const { data } = await apiClient.get<HealthResponse>('/api/health');
    return data;
  },

  // Get all symptoms
  async getSymptoms(): Promise<SymptomsResponse> {
    const { data } = await apiClient.get<SymptomsResponse>('/api/symptoms');
    return data;
  },

  // Predict disease from symptoms
  async predictDisease(symptoms: string[]): Promise<PredictionResponse> {
    const { data } = await apiClient.post<PredictionResponse>('/api/predict', {
      symptoms,
    });
    return data;
  },
};

export default api;
