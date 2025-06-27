import numpy as np
import pandas as pd


def clips_to_cad(df: pd.DataFrame):
    if "csd" in df.columns:
        df["Chain stay back diameter"] = df["csd"]
        df["Chain stay vertical diameter"] = df["csd"]
    if "ssd" in df.columns:
        df["SEATSTAY_HR"] = df["ssd"]
        df["Seat stay bottom diameter"] = df["ssd"]
    if "ttd" in df.columns:
        df["Top tube rear diameter"] = df["ttd"]
        df["Top tube rear dia2"] = df["ttd"]
        df["Top tube front diameter"] = df["ttd"]
        df["Top tube front dia2"] = df["ttd"]
    if "dtd" in df.columns:
        df["Down tube rear diameter"] = df["dtd"]
        df["Down tube rear dia2"] = df["dtd"]
        df["Down tube front diameter"] = df["dtd"]
        df["Down tube front dia2"] = df["dtd"]
    if "RDBSD" in df.columns:
        df["BSD rear"] = df["Wheel diameter rear"] - df["RDBSD"]
        df["ERD rear"] = df["BSD rear"]
    if "FDBSD" in df.columns:
        df["BSD front"] = df["Wheel diameter front"] - df["FDBSD"]
        df["ERD front"] = df["BSD front"]

    Stack = df["Stack"]
    HTL = df["Head tube length textfield"]
    HTLX = df["Head tube lower extension2"]
    HTA = df["Head angle"] * np.pi / 180
    BBD = df["BB textfield"]
    WDR = df["Wheel diameter rear"]
    WDF = df["Wheel diameter front"]
    FBBD = BBD - WDR / 2 + WDF / 2
    DTL = df["DT Length"]
    DTJY = Stack - (HTL - HTLX) * np.sin(HTA)
    DTJX = np.sqrt(DTL ** 2 - DTJY ** 2)
    fork0r = df["FORK0R"]
    Fork_L_plus_HTLX_y = DTJY - FBBD + fork0r * np.cos(HTA)  # Fork L plus HTLX y component
    Fork_L_plus_HTLX = Fork_L_plus_HTLX_y / np.sin(HTA)  # Fork L plus HTLX 
    Fork_L_plus_HTLX_x = Fork_L_plus_HTLX * np.cos(HTA)  # Fork L plus HTLX x component
    bb_to_wheel_x = DTJX + Fork_L_plus_HTLX_x + fork0r * np.sin(HTA)  # x component from BB to wheel
    FCD = np.sqrt(bb_to_wheel_x ** 2 + FBBD ** 2)  # Fork center distance
    Fork_L = Fork_L_plus_HTLX - HTLX  # Fork L
    df["FCD textfield"] = FCD
    df["FORK0L"] = Fork_L
    df["FORK1L"] = Fork_L
    df["FORK2L"] = Fork_L

    df.drop(["DT Length"], axis=1, inplace=True)
    
    r = df["FIRST color R_RGB"].values
    g = df["FIRST color G_RGB"].values
    b = df["FIRST color B_RGB"].values
    
    r = np.round(r).astype(int)
    g = np.round(g).astype(int)
    b = np.round(b).astype(int)

    df.drop("FIRST color R_RGB", axis=1, inplace=True)
    df.drop("FIRST color G_RGB", axis=1, inplace=True)
    df.drop("FIRST color B_RGB", axis=1, inplace=True)
    val = r * (2 ** 16) + g * (2 ** 8) + b - (2 ** 24)
    df["FIRST color sRGB"] = val

    return df.copy()
