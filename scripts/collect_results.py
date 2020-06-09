import os
import argparse


# Each element of KEYWORDS is a tuple of (source, destination) keywords.
KEYWORDS = [
    ("abandoned", "actions+obs_0"),
    ("aberdeen", "actions+obs_1"),
    ("abilities", "actions+obs_2"),
    ("ability", "actions+obs_3"),
    ("aboriginal", "actions+obs_4"),
    ("abortion", "actions_0"),
    ("abraham", "actions_1"),
    ("abroad", "actions_2"),
    ("absence", "actions_3"),
    ("absent", "actions_4"),
    ("absolute", "obs_0"),
    ("absolutely", "obs_1"),
    ("absorption", "obs_2"),
    ("abstract", "obs_3"),
    ("abstracts", "obs_4"),
]
FILE_SUFFIXES = [
    "env_log.txt",
    "settings.json",
]


def main(args):

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
