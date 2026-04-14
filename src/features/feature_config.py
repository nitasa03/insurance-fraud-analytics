# ---------- Target and ID columns ----------
IEEE_TARGET = "isFraud"
IEEE_ID_COLS = ["TransactionID"]

PORTO_TARGET = "target"
PORTO_ID_COLS = ["id"]


# ---------- Columns to exclude from modeling ----------
IEEE_EXCLUDE_FROM_MODEL = IEEE_ID_COLS + [IEEE_TARGET]
PORTO_EXCLUDE_FROM_MODEL = PORTO_ID_COLS + [PORTO_TARGET]


# ---------- IEEE raw columns selected from analysis / EDA ----------
IEEE_SELECTED_RAW_NUMERIC = [
    "TransactionAmt",
    "TransactionDT",
    "card1",
    "D1",
    "D10",
    "D15",
    "V127",
    "V133",
    "V307",
    "V308",
    "V317"
]

IEEE_SELECTED_RAW_CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]


# ---------- Porto raw columns selected from analysis ----------
PORTO_SELECTED_RAW_NUMERIC = [
    "ps_ind_01",
    "ps_ind_03",
    "ps_ind_15",
    "ps_car_11_cat",
    "ps_car_04_cat",
    "ps_car_06_cat",
    "ps_car_13",
    "ps_car_15"
]

PORTO_SELECTED_RAW_BINARY = [
    "ps_ind_06_bin",
    "ps_ind_07_bin",
    "ps_ind_08_bin",
    "ps_ind_09_bin",
    "ps_ind_16_bin",
    "ps_ind_17_bin",
    "ps_ind_18_bin"
]


# ---------- IEEE engineered features planned----------
IEEE_ENGINEERED_FEATURES = [
    "TransactionAmt_log1p",
    "hour_of_day",
    "is_night",
    "email_domain_match",
    "ProductCD_freq",
    "card4_freq",
    "card6_freq",
    "card1_txn_count",
    "card4_amt_mean",
    "amt_deviation_card4"
]


# ---------- Porto engineered features planned ----------
PORTO_ENGINEERED_FEATURES = [
    "vehicle_driver_interaction",
    "high_risk_flag",
    "binary_risk_count"
]


# ---------- Gold output files ----------
IEEE_GOLD_FILE = "ieee_features.csv"
PORTO_GOLD_FILE = "porto_features.csv"