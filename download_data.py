import os
import gdown

def download_reports():

    os.makedirs("data/reports", exist_ok=True)

    files = {
        "apple_Q1_2025.pdf": "1sIDxqH2-kH91Tgb_p4tjISBJpD1UR5UH",
        "apple_Q2_2025.pdf": "1LVn5uCFK7KjeQoqh-jh5p_L1aV_YDetB",
        "infosys_Q1_2025.pdf": "1gi8d78jGrxNup3ghvrp0izt8r4QqCPah",
        "infosys_Q1_2025_press.pdf":"1NrMHNC9-F3SDO_AsyxLaxdrEOtYcDtEQ",
        "infosys_Q2_2025_press.pdf":"1wEGPmKKOWrvfjY4be4VAEw6wBLFhba4m",
        "infosys_Q2_2025.pdf":"1Thw0sUuH7cVgw2fs3xZ1QYXtw6_JkpLS",
        "infosys_Q3_2025_press.pdf":"1Oi-zvgTpoS1E297tFZBafM9Si3o4G_Nz",
        "infosys_Q3_2025.pdf":"1FNwLKnvkvTSXUATViMiWA9v9nDF9m8ix",
        "infosys_Q4_2025_press.pdf":"1_-Rs3E23YBAS8blosUYSOHZGD_QF1Ero",
        "infosys_Q4_2025.pdf":"1-rtbMc-amyL5m9VRWoLWujq9mRBkO1iS",
        }

    for filename, file_id in files.items():
        output = f"data/reports/{filename}"

        if not os.path.exists(output):
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading {filename}...")
            gdown.download(url, output, quiet=False)

    print("All reports downloaded.")