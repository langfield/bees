port=$1
result_dir=$2
ip=root@0.tcp.ngrok.io
source_path=/root/bees/logs/optuna_*.log

# check for arguments
if [ $# -ne 2 ]
then
    echo Must provide two arguments.
    exit
fi

# loop indefinitely
while true
do
    # pull results
    scp -P $port ${ip}:${results_path} $result_dir
    echo scp -P $port ${ip}:${source_path} $result_dir
    if [ $? -eq 0 ]
    then
        echo Copied log at $(date '+%d/%m/%Y %H:%M:%S').
    else
        echo Failed to copy log.
    fi

    # wait for 5 minutes
    sleep 1m
done
