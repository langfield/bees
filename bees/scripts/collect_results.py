import os


# Each element of KEYWORDS is a tuple of (source, destination) keywords.
KEYWORDS = [
    ("abandoned", "accepts"),
    ("aberdeen", "access"),
    ("abilities", "accessed"),
    ("ability", "accessibility"),
    ("aboriginal", "accessible"),
    ("abortion", "accessing"),
    ("abraham", "accessories"),
    ("abroad", "accessory"),
    ("absence", "accident"),
    ("absent", "accidents"),
    ("absolute", "accommodate"),
    ("absolutely", "accommodation"),
]
FILE_SUFFIXES = [
    "env_log.txt",
    "settings.json",
]
PORT_NUMBER = 18473


def main():

    for source_name, dest_name in KEYWORDS:
        for suffix in FILE_SUFFIXES:
            source_path = "root@0.tcp.ngrok.io:/root/pkgs/bees/models/%s_*/*%s" % (source_name, suffix)
            dest_path = "models/%s/%s_%s" % (dest_name, dest_name, suffix)
            command = "scp -P %d %s %s" % (PORT_NUMBER, source_path, dest_path)

            os.system(command)

if __name__ == "__main__":
    main()
