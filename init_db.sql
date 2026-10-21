-- ==========================================================
-- SKEMA DATABASE SQLITE: AI SLAT RIB CONVEYOR (DATA NG)
-- Target: NVIDIA Jetson Orin Nano & Local Development
-- File: AI_Slat_Rib_Conveyor.db
-- ==========================================================

PRAGMA foreign_keys = ON;

-- 1. TABEL MASTER: Inspections
-- Menyimpan header ringkasan setiap siklus bilah/slat conveyor
CREATE TABLE IF NOT EXISTS Inspections (
    InspectionID INTEGER PRIMARY KEY AUTOINCREMENT,
    TraceabilityCode TEXT NOT NULL UNIQUE,
    SlatNumber INTEGER,
    InspectionTime DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    OverallResult TEXT NOT NULL CHECK (OverallResult IN ('OK', 'NG', 'INVALID')),
    OverallSeverity TEXT NOT NULL DEFAULT 'NONE' CHECK (OverallSeverity IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    StatusLifecycle TEXT NOT NULL DEFAULT 'UNACKNOWLEDGED' CHECK (StatusLifecycle IN ('NORMAL', 'UNACKNOWLEDGED', 'ACKNOWLEDGED', 'RESOLVED')),
    PICName TEXT,
    MaintenanceNote TEXT,
    ResolvedAt DATETIME,
    CreatedAt DATETIME NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 2. TABEL DETAIL: Inspection_Detections
-- Menyimpan temuan kondisi tiap komponen dari kamera CAM-01, CAM-02, CAM-03
CREATE TABLE IF NOT EXISTS Inspection_Detections (
    DetectionID INTEGER PRIMARY KEY AUTOINCREMENT,
    InspectionID INTEGER NOT NULL,
    CameraID TEXT NOT NULL CHECK (CameraID IN ('CAM-01', 'CAM-02', 'CAM-03')),
    TargetArea TEXT NOT NULL,
    ComponentType TEXT NOT NULL CHECK (ComponentType IN ('PLAT', 'RIB', 'BOLT', 'PIN', 'LEKUKAN')),
    Side TEXT NOT NULL DEFAULT 'NONE' CHECK (Side IN ('RH', 'LH', 'NONE')),
    Result TEXT NOT NULL CHECK (Result IN ('OK', 'NG', 'INVALID')),
    StatusDetail TEXT NOT NULL CHECK (StatusDetail IN ('NORMAL', 'BENDING', 'MISSING', 'CRACK')),
    Confidence REAL,
    PVFlatness REAL,
    MaxDeviation REAL,
    RawImagePath TEXT,
    AnnotatedPath TEXT,
    CreatedAt DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (InspectionID) REFERENCES Inspections (InspectionID) ON DELETE CASCADE
);

-- 3. TABEL AUDIT: Maintenance_Logs
-- Mencatat audit trail saat operator melakukan Acknowledge atau Mark Resolved
CREATE TABLE IF NOT EXISTS Maintenance_Logs (
    LogID INTEGER PRIMARY KEY AUTOINCREMENT,
    InspectionID INTEGER NOT NULL,
    Action TEXT NOT NULL CHECK (Action IN ('ACKNOWLEDGED', 'RESOLVED')),
    PICName TEXT NOT NULL,
    Notes TEXT,
    LoggedAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (InspectionID) REFERENCES Inspections (InspectionID) ON DELETE CASCADE
);

-- 4. TABEL MASTER REFERENSI: Master_Components
CREATE TABLE IF NOT EXISTS Master_Components (
    ComponentType TEXT PRIMARY KEY,
    ComponentName TEXT NOT NULL,
    DefaultCamera TEXT NOT NULL,
    DefaultSide TEXT NOT NULL
);

INSERT OR IGNORE INTO Master_Components (ComponentType, ComponentName, DefaultCamera, DefaultSide) VALUES
('PLAT', 'Plat / Slat Conveyor', 'CAM-01', 'NONE'),
('RIB', 'Rib / Sirip Conveyor', 'CAM-01', 'NONE'),
('BOLT', 'Bolt Penahan Sisi', 'CAM-02', 'RH'),
('PIN', 'Pin Sambungan Rantai', 'CAM-02', 'RH'),
('LEKUKAN', 'Lekukan Ujung Sisi', 'CAM-02', 'RH');

-- 5. INDEX PERFORMA TINGGI (Optimasi query real-time Jetson)
CREATE INDEX IF NOT EXISTS idx_inspections_time ON Inspections (InspectionTime DESC);
CREATE INDEX IF NOT EXISTS idx_inspections_result ON Inspections (OverallResult);
CREATE INDEX IF NOT EXISTS idx_detections_lookup ON Inspection_Detections (ComponentType, Side, Result);
CREATE INDEX IF NOT EXISTS idx_detections_inspection ON Inspection_Detections (InspectionID);

-- 6. VIEW CEPAT: v_Abnormality_Summary
-- Mempercepat kalkulasi 6 kartu anomali di Frontend
CREATE VIEW IF NOT EXISTS v_Abnormality_Summary AS
SELECT 
    ComponentType,
    Side,
    CameraID,
    COUNT(1) AS TotalNG
FROM Inspection_Detections
WHERE Result = 'NG'
GROUP BY ComponentType, Side, CameraID;
