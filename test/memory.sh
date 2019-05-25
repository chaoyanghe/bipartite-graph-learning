#!/usr/bin/env bash

PID=$1
LOG=./$PID.log
PNG=./$PID.log.png

echo recording memory usage for PID $PID
echo log file: $LOG
echo png file: $PNG

while true; do
    ps --pid $PID -o pid=,%mem=,vsz= >> $LOG
    #gnuplot -e "set term png small size 800,600; set output \"$PNG\"; set ylabel \"VSZ\"; set y2label \"%MEM\"; set ytics nomirror; set y2tics nomirror in; set yrange [*:*]; set y2range [*:*]; plot \"$LOG\" using 3 with lines axes x1y1 title \"VSZ\", \"$LOG\" using 2 with lines axes x1y2 title \"%MEM\""

    sleep 1
done