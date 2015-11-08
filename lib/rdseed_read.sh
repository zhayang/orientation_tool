#!/bin/bash
#input: 1 filename: 2 station name; 3:Channel 4:year 5: date
#day=300
#seedfile=/Volumes/Work/CASCADIA/cascadia.year2.608128.seed
rdseed<<!
$1


d

$2
$3


1
Y
N


$4,$5,00:00:01
$4,$5,23:59:59


quit
 !
