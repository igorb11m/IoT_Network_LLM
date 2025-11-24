import os
import pandas as pd
from scapy.all import PcapReader, IP, TCP, UDP


INPUT_DIR = "./captures_IoT-Sentinel"
OUTPUT_CSV = "iot_features.csv"


def extract_features_from_pcap(pcap_path, label):

    features_list = []
    last_timestamp = 0


    try:
        with PcapReader(pcap_path) as pcap_reader:
            for i, pkt in enumerate(pcap_reader):

                if IP in pkt:


                    pkt_len = len(pkt)


                    current_timestamp = float(pkt.time)
                    if last_timestamp == 0:
                        iat = 0.0
                    else:
                        iat = current_timestamp - last_timestamp
                    last_timestamp = current_timestamp


                    proto = pkt[IP].proto
                    src_port = 0
                    dst_port = 0

                    if TCP in pkt:
                        src_port = pkt[TCP].sport
                        dst_port = pkt[TCP].dport
                    elif UDP in pkt:
                        src_port = pkt[UDP].sport
                        dst_port = pkt[UDP].dport


                    features = {
                        "Packet_Length": pkt_len,
                        "IAT": iat,
                        "Protocol": proto,
                        "Src_Port": src_port,
                        "Dst_Port": dst_port,
                        "TTL": pkt[IP].ttl,
                        "Label": label
                    }
                    features_list.append(features)




    except Exception as e:
        print(f"Error at file {pcap_path}: {e}")

    return features_list


def main():
    all_data = []


    if not os.path.exists(INPUT_DIR):
        print(f"Error: directory not found {INPUT_DIR}")
        return

    print("Processing data ...")

    for device_name in os.listdir(INPUT_DIR):
        device_path = os.path.join(INPUT_DIR, device_name)

        if os.path.isdir(device_path):
            print(f"Processing device: {device_name}")

            for file in os.listdir(device_path):
                if file.endswith(".pcap") or file.endswith(".pcapng"):
                    file_path = os.path.join(device_path, file)


                    data = extract_features_from_pcap(file_path, label=device_name)
                    all_data.extend(data)
                    print(f"  -> Processed {file}: {len(data)} packets.")


    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\Saved to file: {OUTPUT_CSV}")
        print(f"Number of samples: {len(df)}")
        print(df.head())
    else:
        print("Data for processing not found.")


if __name__ == "__main__":
    main()