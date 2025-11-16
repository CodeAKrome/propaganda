#!/usr/bin/perl
use strict;
use warnings;

my $replace = 0;
while (my $line = <STDIN>) {
    $replace = 0 if $line =~ /^---/;

    if ($replace) {
        # drop whole <script>…</script> blocks
        next if $line =~ /<script/i .. $line =~ m!</script>!i;

        # drop lines with fewer than 5 words
        my $words = () = $line =~ /\S+/g;
        next if $words < 5;
    }

    $replace = 1 if $line =~ /^Text:/;
    print $line;
}
