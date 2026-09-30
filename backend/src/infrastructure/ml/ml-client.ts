import { AppError } from "@/lib/errors";

export interface PredictRiskInput {
  wellId: string;
  depthMd: number;
  formation?: string;
  lithology?: string;
  ropMhr?: number;
  wobKlbs?: number;
  rpm?: number;
  torqueKftlb?: number;
  sppPsi?: number;
  flowRateGpm?: number;
  mudWeightPpg?: number;
  ecdPpg?: number;
  offsetWellsCountInRadius?: number;
  maxOffsetSimilarity?: number;
  offsetMudLossCount?: number;
  offsetStuckPipeCount?: number;
  offsetKickCount?: number;
  offsetTorqueSpikeCount?: number;
  nearestHazardDistanceKm?: number;
  offsetHazardDensity?: number;
}

export interface RiskPredictionResult {
  well_id: string;
  depth_md: number;
  risk_type: string;
  probability: number;
  level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  model_version: string;
  top_contributing_features: Array<{ feature: string; importance: number }>;
  mitigation_recommendation: string;
}

export interface RAGQueryFilterInput {
  wellId?: string;
  formation?: string;
  eventType?: string;
  minDepth?: number;
  maxDepth?: number;
}

export interface RAGSourceChunk {
  document_id: string;
  page: number;
  section: string;
  text_snippet: string;
  well_id?: string | null;
  formation?: string | null;
  depth_md?: number | null;
  score: number;
}

export interface RAGQueryResponse {
  query: string;
  answer: string;
  confidence: number;
  sources: RAGSourceChunk[];
}

export class MLClient {
  private readonly baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = (
      baseUrl ||
      process.env.ML_SERVICE_URL ||
      process.env.AI_ML_SERVICE_URL ||
      "http://127.0.0.1:8000"
    ).replace(/\/$/, "");
  }

  async checkHealth(): Promise<{
    status: string;
    risk_model_loaded: boolean;
    rag_assistant_loaded: boolean;
    rag_chunks_indexed: number;
  }> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, {
        method: "GET",
        headers: { Accept: "application/json" },
      });
      if (!res.ok) {
        throw new Error(`ML Health Check failed: ${res.statusText}`);
      }
      return await res.json();
    } catch (err: any) {
      throw AppError.internal(`ML service unreachable: ${err.message}`);
    }
  }

  async predictMudLossRisk(input: PredictRiskInput): Promise<RiskPredictionResult> {
    const payload = {
      well_id: input.wellId,
      depth_md: input.depthMd,
      formation: input.formation,
      lithology: input.lithology || "Porous Sandstone / Siltstone",
      rop_mhr: input.ropMhr ?? 12.0,
      wob_klbs: input.wobKlbs ?? 24.0,
      rpm: input.rpm ?? 110.0,
      torque_kftlb: input.torqueKftlb ?? 8.0,
      spp_psi: input.sppPsi ?? 2200.0,
      flow_rate_gpm: input.flowRateGpm ?? 550.0,
      mud_weight_ppg: input.mudWeightPpg ?? 9.8,
      ecd_ppg: input.ecdPpg ?? 10.2,
      offset_wells_count_in_radius: input.offsetWellsCountInRadius,
      max_offset_similarity: input.maxOffsetSimilarity,
      offset_mud_loss_count: input.offsetMudLossCount,
      offset_stuck_pipe_count: input.offsetStuckPipeCount,
      offset_kick_count: input.offsetKickCount,
      offset_torque_spike_count: input.offsetTorqueSpikeCount,
      nearest_hazard_distance_km: input.nearestHazardDistanceKm,
      offset_hazard_density: input.offsetHazardDensity,
    };

    try {
      const res = await fetch(`${this.baseUrl}/api/v1/predict/mud-loss`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`ML prediction failed (${res.status}): ${errorText}`);
      }

      return await res.json();
    } catch (err: any) {
      if (err instanceof AppError) throw err;
      throw AppError.internal(`ML risk prediction failed: ${err.message}`);
    }
  }

  async queryRAG(
    query: string,
    filters?: RAGQueryFilterInput,
    topK: number = 4,
  ): Promise<RAGQueryResponse> {
    const payload = {
      query,
      well_id: filters?.wellId,
      formation: filters?.formation,
      event_type: filters?.eventType,
      min_depth: filters?.minDepth,
      max_depth: filters?.maxDepth,
      top_k: topK,
    };

    try {
      const res = await fetch(`${this.baseUrl}/api/v1/rag/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`ML RAG query failed (${res.status}): ${errorText}`);
      }

      return await res.json();
    } catch (err: any) {
      if (err instanceof AppError) throw err;
      throw AppError.internal(`ML RAG query failed: ${err.message}`);
    }
  }
}

export const mlClient = new MLClient();
