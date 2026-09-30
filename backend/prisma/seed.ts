import {
  PrismaClient,
  Role,
  WellStatus,
  DocumentType,
  IngestionStatus,
  EventType,
  EventSeverity,
  ReviewStatus,
  AlertStatus,
} from "@prisma/client";
import bcrypt from "bcryptjs";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { parse } from "csv-parse/sync";

const prisma = new PrismaClient();

const ML_ROOT = path.resolve(__dirname, "../../ml");
const RAW_DOCS_DIR = path.join(ML_ROOT, "data/raw/good_data");
const PROCESSED_DIR = path.join(ML_ROOT, "data/processed/good_data_processed");
const WELLS_CSV_PATH = path.join(PROCESSED_DIR, "historical_dataset/wells.csv");
const EVENTS_CSV_PATH = path.join(PROCESSED_DIR, "historical_dataset/events.csv");
const RISK_DATASET_CSV_PATH = path.join(ML_ROOT, "data/processed/risk_prediction_dataset.csv");

// Regional fallback coordinates for Upper Assam wells without explicit GPS in DDR headers
const REGIONAL_COORDINATES: Record<string, { lat: number; lon: number; field: string; plannedDepth: number }> = {
  "DLJ-07": { lat: 27.5072, lon: 95.0751, field: "Duliajan", plannedDepth: 3500 },
  "DLJ-12": { lat: 27.5130, lon: 95.1486, field: "Duliajan", plannedDepth: 3400 },
  "DLJ-18": { lat: 27.4612, lon: 95.0595, field: "Duliajan", plannedDepth: 3350 },
  "DLJ-21": { lat: 27.4560, lon: 95.1349, field: "Duliajan", plannedDepth: 3100 },
  "HGJ-02": { lat: 27.3794, lon: 95.3512, field: "Hugrijan", plannedDepth: 4100 },
  "NHK-07": { lat: 27.4013, lon: 95.2012, field: "Naharkatiya", plannedDepth: 3300 },
  "NHR-04": { lat: 27.4150, lon: 95.1850, field: "UpperAssam", plannedDepth: 3600 },
  "W-076":  { lat: 27.4350, lon: 95.1200, field: "UpperAssam", plannedDepth: 3200 },
  "W-087":  { lat: 27.4600, lon: 95.0600, field: "UpperAssam", plannedDepth: 3700 },
  "W-093":  { lat: 27.4420, lon: 95.1100, field: "UpperAssam", plannedDepth: 3550 },
  "W-098":  { lat: 27.4450, lon: 95.1200, field: "UpperAssam", plannedDepth: 4100 },
  "W-105":  { lat: 27.3800, lon: 95.0900, field: "Moran", plannedDepth: 3650 },
};

function inferDocumentType(filename: string): DocumentType {
  const upper = filename.toUpperCase();
  if (upper.startsWith("DDR")) return DocumentType.DDR;
  if (upper.startsWith("WCR")) return DocumentType.WCR;
  if (upper.includes("MUDLOG") || upper.includes("MUD_LOG")) return DocumentType.MUD_LOG;
  if (upper.includes("CEMENTING")) return DocumentType.CEMENTING_REPORT;
  return DocumentType.WCR;
}

function mapEventType(typeStr: string): EventType {
  const clean = (typeStr || "").trim().toUpperCase();
  switch (clean) {
    case "MUD_LOSS":
    case "LOST_CIRCULATION":
      return EventType.MUD_LOSS;
    case "KICK":
    case "OVERPRESSURE":
    case "WELL_CONTROL":
      return EventType.KICK;
    case "STUCK_PIPE":
    case "FISHING":
      return EventType.STUCK_PIPE;
    case "TORQUE_SPIKE":
    case "PRESSURE_SPIKE":
      return EventType.TORQUE_SPIKE;
    case "CEMENTING_FAILURE":
      return EventType.CEMENTING_FAILURE;
    default:
      return EventType.MUD_LOSS;
  }
}

function mapEventSeverity(sevStr: string | null | undefined): EventSeverity {
  const clean = (sevStr || "").trim().toUpperCase();
  switch (clean) {
    case "CRITICAL":
      return EventSeverity.CRITICAL;
    case "HIGH":
      return EventSeverity.HIGH;
    case "MEDIUM":
      return EventSeverity.MEDIUM;
    case "LOW":
      return EventSeverity.LOW;
    default:
      return EventSeverity.MEDIUM;
  }
}

async function main() {
  console.log("==================================================================");
  console.log("🌱 Starting NWIS database seeding from ML processed dataset...");
  console.log(`ML Root: ${ML_ROOT}`);
  console.log("==================================================================");

  // 1. Seed Demo & Engineer Users
  const passwordHash = await bcrypt.hash("Password123!", 10);

  const demoUser = await prisma.user.upsert({
    where: { email: "demo@nwis.gov.in" },
    update: { role: Role.ADMIN, isActive: true },
    create: {
      email: "demo@nwis.gov.in",
      passwordHash,
      role: Role.ADMIN,
      isActive: true,
    },
  });

  const engineerUser = await prisma.user.upsert({
    where: { email: "engineer@nwis.gov.in" },
    update: { role: Role.DRILLING_ENGINEER, isActive: true },
    create: {
      email: "engineer@nwis.gov.in",
      passwordHash,
      role: Role.DRILLING_ENGINEER,
      isActive: true,
    },
  });

  console.log(`👤 Users seeded: demo@nwis.gov.in (${demoUser.role}), engineer@nwis.gov.in (${engineerUser.role})`);

  // 2. Read and Seed Wells from ml/data/processed/good_data_processed/historical_dataset/wells.csv
  if (!fs.existsSync(WELLS_CSV_PATH)) {
    throw new Error(`wells.csv not found at ${WELLS_CSV_PATH}`);
  }

  const rawWellsCsv = fs.readFileSync(WELLS_CSV_PATH, "utf-8");
  const wellsRecords: Array<{
    well_id: string;
    field: string;
    latitude: string;
    longitude: string;
    source_documents: string;
    source_pages: string;
  }> = parse(rawWellsCsv, { columns: true, skip_empty_lines: true });

  console.log(`📊 Found ${wellsRecords.length} wells in ML historical dataset.`);

  // Purge any legacy non-ML wells so database strictly contains ML data
  const mlWellIds = wellsRecords.map((r) => r.well_id.trim());
  await prisma.drillingEvent.deleteMany({ where: { well: { wellId: { notIn: mlWellIds } } } });
  await prisma.alert.deleteMany({ where: { well: { wellId: { notIn: mlWellIds } } } });
  await prisma.telemetryReading.deleteMany({ where: { well: { wellId: { notIn: mlWellIds } } } });
  await prisma.document.deleteMany({ where: { well: { wellId: { notIn: mlWellIds } } } });
  await prisma.formation.deleteMany({ where: { well: { wellId: { notIn: mlWellIds } } } });
  await prisma.well.deleteMany({ where: { wellId: { notIn: mlWellIds } } });

  const wellDbMap: Record<string, any> = {};

  for (const r of wellsRecords) {
    const wellId = r.well_id.trim();
    const regional = REGIONAL_COORDINATES[wellId] || { lat: 27.45, lon: 95.10, field: "UpperAssam", plannedDepth: 3500 };

    const lat = r.latitude && !isNaN(parseFloat(r.latitude)) ? parseFloat(r.latitude) : regional.lat;
    const lon = r.longitude && !isNaN(parseFloat(r.longitude)) ? parseFloat(r.longitude) : regional.lon;
    const field = r.field ? r.field.trim() : regional.field;
    const status = wellId === "DLJ-07" ? WellStatus.DRILLING : WellStatus.COMPLETED;
    const plannedDepth = regional.plannedDepth;

    const well = await prisma.well.upsert({
      where: { wellId },
      update: {
        name: `OIL-${wellId}`,
        field,
        latitude: lat,
        longitude: lon,
        status,
        plannedDepthMd: plannedDepth,
        plannedDepthTvd: plannedDepth - 180,
      },
      create: {
        wellId,
        name: `OIL-${wellId}`,
        field,
        latitude: lat,
        longitude: lon,
        spudDate: new Date("2021-01-10T00:00:00Z"),
        plannedDepthMd: plannedDepth,
        plannedDepthTvd: plannedDepth - 180,
        status,
      },
    });

    wellDbMap[wellId] = well;

    // Seed regional stratigraphy formations from ml/src/features/stratigraphy.py
    const regionalFormations = [
      { name: "Alluvium / Girujan Clay", top: 0, base: 1500, lithology: "Claystone / Sand" },
      { name: "Tipam Sandstone", top: 1500, base: 2800, lithology: "Porous Sandstone / Siltstone" },
      { name: "Barail", top: 2800, base: 3600, lithology: "Coal / Shale / Sandstone Sequence" },
      { name: "Kopili", top: 3100, base: 3700, lithology: "Overpressured Fissile Shale" },
      { name: "Sylhet Limestone", top: 3700, base: 4500, lithology: "Fractured Vugular Limestone" },
    ];

    for (const fmt of regionalFormations) {
      const existingFmt = await prisma.formation.findFirst({
        where: { wellId: well.id, name: fmt.name },
      });

      if (!existingFmt) {
        await prisma.formation.create({
          data: {
            wellId: well.id,
            name: fmt.name,
            topMd: fmt.top,
            bottomMd: fmt.base,
            lithology: fmt.lithology,
          },
        });
      }
    }
  }

  console.log(`✅ Seeded ${Object.keys(wellDbMap).length} wells with Upper Assam stratigraphy.`);

  // 3. Read and Seed Documents from ml/data/raw/good_data
  const docDbMap: Record<string, any> = {};

  if (fs.existsSync(RAW_DOCS_DIR)) {
    const rawFiles = fs.readdirSync(RAW_DOCS_DIR).filter((f) => f.toLowerCase().endsWith(".pdf"));
    console.log(`📄 Found ${rawFiles.length} raw PDF documents in ${RAW_DOCS_DIR}`);

    for (const filename of rawFiles) {
      const filePath = path.join(RAW_DOCS_DIR, filename);
      const stat = fs.statSync(filePath);
      const fileBuffer = fs.readFileSync(filePath);
      const hash = crypto.createHash("sha256").update(fileBuffer).digest("hex");
      const docType = inferDocumentType(filename);

      // Infer associated well
      let matchedWellId: string | null = null;
      for (const wid of Object.keys(wellDbMap)) {
        const cleanWid = wid.replace("-", "");
        const cleanFn = filename.replace("-", "").replace("_", "");
        if (cleanFn.includes(cleanWid)) {
          matchedWellId = wid;
          break;
        }
      }
      if (!matchedWellId) {
        if (filename.includes("NHR-04") || filename.includes("Loss_Study")) matchedWellId = "NHR-04";
        else if (filename.includes("W076")) matchedWellId = "W-076";
        else if (filename.includes("W087")) matchedWellId = "W-087";
        else if (filename.includes("W093")) matchedWellId = "W-093";
        else if (filename.includes("W098")) matchedWellId = "W-098";
        else if (filename.includes("W105")) matchedWellId = "W-105";
        else matchedWellId = "DLJ-07";
      }

      const well = wellDbMap[matchedWellId] || wellDbMap["DLJ-07"];

      const existingDoc = await prisma.document.findFirst({
        where: { filename },
      });

      const doc = existingDoc
        ? await prisma.document.update({
            where: { id: existingDoc.id },
            data: {
              documentType: docType,
              mimeType: "application/pdf",
              fileSize: stat.size,
              fileHash: hash,
              storageKey: `documents/${filename}`,
              uploadedBy: engineerUser.id,
              ingestionStatus: IngestionStatus.COMPLETED,
            },
          })
        : await prisma.document.create({
            data: {
              wellId: well.id,
              filename,
              mimeType: "application/pdf",
              documentType: docType,
              fileSize: stat.size,
              fileHash: hash,
              storageKey: `documents/${filename}`,
              uploadedBy: engineerUser.id,
              ingestionStatus: IngestionStatus.COMPLETED,
            },
          });

      docDbMap[filename] = doc;
    }
  }

  console.log(`✅ Seeded ${Object.keys(docDbMap).length} technical documents from ML raw data.`);

  // 4. Read and Seed Drilling Events from ml/data/processed/good_data_processed/historical_dataset/events.csv
  if (!fs.existsSync(EVENTS_CSV_PATH)) {
    throw new Error(`events.csv not found at ${EVENTS_CSV_PATH}`);
  }

  const rawEventsCsv = fs.readFileSync(EVENTS_CSV_PATH, "utf-8");
  const eventsRecords: Array<{
    event_id: string;
    well_id: string;
    event_type: string;
    depth_md: string;
    depth_tvd: string;
    formation: string;
    severity: string;
    description: string;
    cause: string;
    mitigation: string;
    outcome: string;
    source_document: string;
    source_page: string;
    confidence: string;
    npt_hours: string;
    source_section: string;
    extraction_model: string;
    evidence: string;
  }> = parse(rawEventsCsv, { columns: true, skip_empty_lines: true });

  console.log(`🔍 Found ${eventsRecords.length} extracted drilling events in ML dataset.`);

  let eventCount = 0;
  for (const ev of eventsRecords) {
    const well = wellDbMap[ev.well_id] || wellDbMap["DLJ-07"];
    const doc = docDbMap[ev.source_document] || Object.values(docDbMap)[0];

    if (!doc) {
      console.warn(`[Seed] Skipping event ${ev.event_id}: no document found`);
      continue;
    }

    const eventType = mapEventType(ev.event_type);
    const severity = mapEventSeverity(ev.severity);
    let depthMd = ev.depth_md && !isNaN(parseFloat(ev.depth_md)) ? parseFloat(ev.depth_md) : null;
    if (depthMd === null) {
      const match = (ev.description || "").match(/(\d{3,4}(?:\.\d+)?)\s*m/i) ||
                    (ev.evidence || "").match(/(\d{3,4}(?:\.\d+)?)\s*m/i);
      depthMd = match ? parseFloat(match[1]) : 3420.0;
    }
    const depthTvd = ev.depth_tvd && !isNaN(parseFloat(ev.depth_tvd)) ? parseFloat(ev.depth_tvd) : (depthMd - 20);
    const confidence = ev.confidence && !isNaN(parseFloat(ev.confidence)) ? parseFloat(ev.confidence) : 0.85;
    const nptHours = ev.npt_hours && !isNaN(parseFloat(ev.npt_hours)) ? parseFloat(ev.npt_hours) : null;

    let parsedEvidence: any = null;
    if (ev.evidence) {
      try {
        parsedEvidence = JSON.parse(ev.evidence);
      } catch {
        parsedEvidence = [{ field: "description", original_text: ev.description }];
      }
    }

    const existingEvent = await prisma.drillingEvent.findFirst({
      where: { mlEventId: ev.event_id },
    });

    if (existingEvent) {
      await prisma.drillingEvent.update({
        where: { id: existingEvent.id },
        data: {
          wellId: well.id,
          sourceDocumentId: doc.id,
          eventType,
          severity,
          depthMd,
          depthTvd,
          formation: ev.formation || "Barail",
          description: ev.description || `${eventType} observed at ${depthMd}m`,
          cause: ev.cause || null,
          mitigation: ev.mitigation || null,
          outcome: ev.outcome || null,
          sourcePage: parseInt(ev.source_page || "1", 10),
          extractionConfidence: confidence,
          reviewStatus: ReviewStatus.APPROVED,
          reviewedBy: engineerUser.id,
          reviewedAt: new Date(),
          nptHours,
          sourceSection: ev.source_section || "DRILLING EVENTS / REMARKS",
          extractionModel: ev.extraction_model || "nwis-rules-baseline-v1",
          evidence: parsedEvidence,
        },
      });
    } else {
      await prisma.drillingEvent.create({
        data: {
          wellId: well.id,
          sourceDocumentId: doc.id,
          mlEventId: ev.event_id,
          eventType,
          severity,
          depthMd,
          depthTvd,
          formation: ev.formation || "Barail",
          description: ev.description || `${eventType} observed at ${depthMd}m`,
          cause: ev.cause || null,
          mitigation: ev.mitigation || null,
          outcome: ev.outcome || null,
          sourcePage: parseInt(ev.source_page || "1", 10),
          extractionConfidence: confidence,
          reviewStatus: ReviewStatus.APPROVED,
          reviewedBy: engineerUser.id,
          reviewedAt: new Date(),
          nptHours,
          sourceSection: ev.source_section || "DRILLING EVENTS / REMARKS",
          extractionModel: ev.extraction_model || "nwis-rules-baseline-v1",
          evidence: parsedEvidence,
        },
      });
    }
    eventCount++;
  }

  console.log(`✅ Seeded ${eventCount} approved drilling events with ML provenance.`);

  // 5. Seed Telemetry for Active Well DLJ-07 from risk_prediction_dataset.csv
  const activeWell = wellDbMap["DLJ-07"];
  if (activeWell && fs.existsSync(RISK_DATASET_CSV_PATH)) {
    const rawRiskCsv = fs.readFileSync(RISK_DATASET_CSV_PATH, "utf-8");
    const riskRecords: Array<{
      well_id: string;
      depth_md: string;
      formation: string;
      rop_mhr: string;
      wob_klbs: string;
      rpm: string;
      torque_kftlb: string;
      spp_psi: string;
      flow_rate_gpm: string;
      mud_weight_ppg: string;
      ecd_ppg: string;
    }> = parse(rawRiskCsv, { columns: true, skip_empty_lines: true });

    // Filter telemetry for DLJ-07
    const dljTelemetry = riskRecords.filter((r) => r.well_id === "DLJ-07").slice(-30); // Last 30 readings
    console.log(`📡 Seeding ${dljTelemetry.length} real-time telemetry measurements for active well DLJ-07...`);

    // Clean old alerts and telemetry for this well
    await prisma.alert.deleteMany({
      where: { wellId: activeWell.id },
    });
    await prisma.telemetryReading.deleteMany({
      where: { wellId: activeWell.id },
    });

    let seq = BigInt(1000);
    const baseTime = new Date("2026-09-28T14:00:00Z").getTime();

    for (let i = 0; i < dljTelemetry.length; i++) {
      const row = dljTelemetry[i];
      const depth = parseFloat(row.depth_md);
      const timestamp = new Date(baseTime + i * 60 * 1000); // 1 minute intervals

      await prisma.telemetryReading.create({
        data: {
          wellId: activeWell.id,
          sourceId: "RIG-SENSORS-DLJ-07",
          sequenceNumber: seq++,
          timestamp,
          depthMd: depth,
          depthTvd: depth - 20,
          rateOfPenetration: parseFloat(row.rop_mhr),
          hookLoad: parseFloat(row.wob_klbs),
          rotaryRpm: parseFloat(row.rpm),
          surfaceTorque: parseFloat(row.torque_kftlb) * 1000, // convert kft-lb to ft-lb
          standpipePressure: parseFloat(row.spp_psi),
          flowRateIn: parseFloat(row.flow_rate_gpm),
          flowRateOut: parseFloat(row.flow_rate_gpm) - (depth >= 3110 ? 40 : 0), // flow discrepancy near kick
          mudDensity: parseFloat(row.mud_weight_ppg),
          metadata: {
            wob_klbs: parseFloat(row.wob_klbs),
            ecd_ppg: parseFloat(row.ecd_ppg),
            formation: row.formation,
          },
        },
      });
    }
    console.log(`✅ Seeded telemetry stream for active well DLJ-07.`);

    // 6. Seed Deterministic Alert Rules and Active Alerts on DLJ-07
    const rule = await prisma.alertRule.upsert({
      where: { ruleCode: "RULE_FLOW_DISCREPANCY_KOPILI" },
      update: {},
      create: {
        ruleCode: "RULE_FLOW_DISCREPANCY_KOPILI",
        name: "Kopili Shale Influx / Kick Rule",
        description: "Monitors flow delta discrepancies between flowRateIn and flowRateOut",
        eventType: EventType.KICK,
        isEnabled: true,
      },
    });

    const ruleVersion = await prisma.alertRuleVersion.upsert({
      where: { id: "rule-ver-kopili-kick-01" },
      update: {},
      create: {
        id: "rule-ver-kopili-kick-01",
        ruleId: rule.id,
        version: 1,
        isActive: true,
        severity: EventSeverity.CRITICAL,
        conditions: {
          type: "DELTA",
          metricA: "flowRateIn",
          metricB: "flowRateOut",
          operator: ">",
          threshold: 30,
          unit: "gpm",
          consecutiveReadings: 2,
        },
        description: "Flow out exceeds flow in by 30 gpm sustained for 2 minutes",
      },
    });

    const latestTelemetry = await prisma.telemetryReading.findFirst({
      where: { wellId: activeWell.id },
      orderBy: { sequenceNumber: "desc" },
    });

    if (latestTelemetry) {
      await prisma.alert.upsert({
        where: { id: "alert-kick-dlj07-001" },
        update: {
          status: AlertStatus.ACTIVE,
          explanation: "Critical flow discrepancy detected at 3,112 m MD in Kopili formation. Pit volume influx observed.",
        },
        create: {
          id: "alert-kick-dlj07-001",
          wellId: activeWell.id,
          ruleVersionId: ruleVersion.id,
          telemetryReadingId: latestTelemetry.id,
          alertType: "KICK",
          severity: EventSeverity.CRITICAL,
          status: AlertStatus.ACTIVE,
          triggeredAt: new Date("2026-09-28T14:29:00Z"),
          explanation: "Critical flow discrepancy detected at 3,112 m MD in Kopili formation. Pit volume influx observed.",
          evidence: ["Pit volume gain 18 bbl", "SIDPP 620 psi", "SICP 780 psi", "Gas peak 1450 units"],
        },
      });
      console.log(`🚨 Seeded active CRITICAL KICK alert on active well DLJ-07.`);
    }
  }

  console.log("==================================================================");
  console.log("🎉 NWIS database seeding completed successfully from ML data!");
  console.log("==================================================================");
}

main()
  .catch((e) => {
    console.error("❌ Seeding failed:", e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
