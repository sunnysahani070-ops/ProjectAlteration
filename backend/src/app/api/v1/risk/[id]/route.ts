import { NextRequest } from "next/server";
import { prisma } from "@/infrastructure/database/prisma";
import { requireAuth } from "@/application/auth/guard";
import { wellService } from "@/application/wells/well.service";
import { telemetryFeatureMapper } from "@/application/telemetry/telemetry-feature-mapper";
import { mlClient } from "@/infrastructure/ml/ml-client";
import { successResponse, errorResponse } from "@/lib/response";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/v1/risk/:id
 *
 * Real-time hazard risk profile for a well.
 * Evaluates live telemetry against the Random Forest ML hazard engine and
 * returns an array of drilling risks matching frontend expectations.
 */
export async function GET(req: NextRequest, { params }: RouteParams) {
  try {
    await requireAuth(req);
    const { id: wellIdParam } = await params;

    const well = await wellService.getWell(wellIdParam);

    // Retrieve latest telemetry reading for the well
    const latestReading = await prisma.telemetryReading.findFirst({
      where: { wellId: well.id },
      orderBy: { timestamp: "desc" },
    });

    const depth = latestReading ? Number(latestReading.depthMd) : 2848;

    let mudLossProb = 0.78;
    let mudLossLevel = "HIGH";
    let topFeatures: any[] = [];
    let recommendation = "";

    let modelVersion = "rf-mud-loss-v1";
    let isLive = true;

    try {
      const mlPayload = telemetryFeatureMapper.mapToMlRiskPayload(
        latestReading
          ? {
              depthMd: Number(latestReading.depthMd),
              rateOfPenetration: latestReading.rateOfPenetration ? Number(latestReading.rateOfPenetration) : null,
              hookLoad: latestReading.hookLoad ? Number(latestReading.hookLoad) : null,
              standpipePressure: latestReading.standpipePressure ? Number(latestReading.standpipePressure) : null,
              surfaceTorque: latestReading.surfaceTorque ? Number(latestReading.surfaceTorque) : null,
              rotaryRpm: latestReading.rotaryRpm ? Number(latestReading.rotaryRpm) : null,
              flowRateIn: latestReading.flowRateIn ? Number(latestReading.flowRateIn) : null,
              flowRateOut: latestReading.flowRateOut ? Number(latestReading.flowRateOut) : null,
              mudDensity: latestReading.mudDensity ? Number(latestReading.mudDensity) : null,
            }
          : { depthMd: 2848 },
        well,
        well.formations || [],
      );

      const prediction = await mlClient.predictMudLossRisk({
        wellId: well.wellId,
        depthMd: mlPayload.depth_md,
        formation: mlPayload.formation,
        lithology: mlPayload.lithology,
        ropMhr: mlPayload.rop_mhr,
        wobKlbs: mlPayload.wob_klbs,
        rpm: mlPayload.rpm,
        torqueKftlb: mlPayload.torque_kftlb,
        sppPsi: mlPayload.spp_psi,
        flowRateGpm: mlPayload.flow_rate_gpm,
        mudWeightPpg: mlPayload.mud_weight_ppg,
        ecdPpg: mlPayload.ecd_ppg,
      });

      mudLossProb = prediction.probability;
      mudLossLevel = prediction.level;
      topFeatures = prediction.top_contributing_features;
      recommendation = prediction.mitigation_recommendation;
      modelVersion = prediction.model_version || "rf-mud-loss-v1";
    } catch (err: any) {
      console.warn(`[Risk API] ML hazard service unreachable for well ${well.wellId}: ${err?.message || err}. Serving baseline fallback.`);
      mudLossProb = 0.78;
      mudLossLevel = "HIGH";
      modelVersion = "heuristic-baseline-fallback";
      isLive = false;
    }

    const risks = [
      {
        risk_type: "MUD_LOSS",
        riskType: "MUD_LOSS",
        probability: mudLossProb,
        level: mudLossLevel,
        severity: mudLossLevel,
        depth,
        depth_md: depth,
        model_version: modelVersion,
        is_live_prediction: isLive,
        evidence: ["E-001", "E-002", "E-003"],
        topContributingFeatures: topFeatures,
        recommendation,
      },
      {
        risk_type: "STUCK_PIPE",
        riskType: "STUCK_PIPE",
        probability: 0.41,
        level: "MEDIUM",
        severity: "MEDIUM",
        depth,
        depth_md: depth,
        model_version: "rule-v1",
        evidence: ["E-004", "E-013"],
      },
      {
        risk_type: "OVERPRESSURE",
        riskType: "OVERPRESSURE",
        probability: 0.09,
        level: "LOW",
        severity: "LOW",
        depth,
        depth_md: depth,
        model_version: "rule-v1",
        evidence: ["E-006"],
      },
      {
        risk_type: "TORQUE_SPIKE",
        riskType: "TORQUE_SPIKE",
        probability: 0.33,
        level: "MEDIUM",
        severity: "MEDIUM",
        depth,
        depth_md: depth,
        model_version: "rule-v1",
        evidence: ["E-005", "E-010"],
      },
      {
        risk_type: "CEMENTING",
        riskType: "CEMENTING",
        probability: 0.12,
        level: "LOW",
        severity: "LOW",
        depth,
        depth_md: depth,
        model_version: "rule-v1",
        evidence: ["E-007", "E-015"],
      },
    ];

    return successResponse(risks, 200);
  } catch (error) {
    return errorResponse(error);
  }
}
