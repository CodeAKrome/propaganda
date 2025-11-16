#!/usr/bin/env perl

while (<>) {
    chomp;
    $in = $_;
    s/^.*\/(\w+)\.md/\1.mp3/;
    $cmd="./ttskokoro.py $in mp3/$_\n";
    print $cmd;
}
