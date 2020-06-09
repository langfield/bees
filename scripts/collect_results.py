import os
import argparse


# Each element of KEYWORDS is a tuple of (source, destination) keywords.
KEYWORDS = [
    ("abandoned", "obs_tabular_0"),
    ("aberdeen", "obs_tabular_1"),
    ("abilities", "obs_tabular_2"),
    ("ability", "obs_tabular_3"),
    ("aboriginal", "obs_raw_0"),
    ("abortion", "obs_raw_1"),
    ("abraham", "obs_raw_2"),
    ("abroad", "obs_raw_3"),
    ("absence", "action+obs_tabular_0"),
    ("absent", "action+obs_tabular_1"),
    ("absolute", "action+obs_tabular_2"),
    ("absolutely", "action+obs_tabular_3"),
    ("absorption", "action+obs_raw_0"),
    ("abstract", "action+obs_raw_1"),
    ("abstracts", "action+obs_raw_2"),
    ("academic", "action+obs_raw_3"),
]
FILE_SUFFIXES = [
    "env_log.txt",
    "settings.json",
]


def main():

    for source_name, dest_name in KEYWORDS:
        for suffix in FILE_SUFFIXES:
            source_path = "root@0.tcp.ngrok.io:/root/pkgs/bees/models/%s_*/*%s" % (
                source_name,
                suffix,
            )
            dest_path = "models/%s/%s_%s" % (dest_name, dest_name, suffix)
            if not os.path.isdir(os.path.dirname(dest_path)):
                os.makedirs(os.path.dirname(dest_path))

            command = "sshpass -p '%s' | scp -P %d %s %s" % (
                args.password,
                args.port_number,
                source_path,
                dest_path,
            )
            os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("password", type=str, help="Password for scp connection.")
    parser.add_argument("port_number", type=int, help="Port number of scp connection.")
    args = parser.parse_args()
    main(args)
