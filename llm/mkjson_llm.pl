#!/usr/bin/env perl
use strict;
use warnings;

# Make custom ollama LLM config file to set temperaure to 0.0 and output JSON.
#
# Usage:
#   echo "image_name" | mkjson_llm.pl
#   or provide multiple image names via stdin, one per line
#   image_name: defaults to 'llama3.1:8b' if no input provided

my $temperature = 0.0;

my $system = <<'SYS';
Output valid JSON only.

Do not reply any other way but valid JSON.
SYS

# Read from stdin line by line
while (my $line = <STDIN>) {
    chomp $line;
    $line =~ s/^\s+|\s+$//g;  # Trim whitespace
    next if $line eq '';      # Skip empty lines
    
    my $image = $line || 'llama3.1:8b';
    
    # Sanitize image name: keep only alphanumerics for filename
    my $basefile = $image;
    $basefile =~ s/[^a-zA-Z0-9]//g;
    
    # Create outfile name based on image
    my $outfile = "json-${basefile}";
    my $cfg_file = "${outfile}.cfg";
    
    open my $fh, '>', $cfg_file or do {
        warn "Cannot open '$cfg_file' for writing: $!\n";
        next;
    };
    
    print $fh "FROM $image\n\n";
    printf $fh "PARAMETER temperature %.1f\n\n", $temperature;
    print $fh "SYSTEM \"\"\"\n";
    print $fh $system, "\n";
    print $fh "\"\"\"\n";
    
    close $fh;
    
    print "Prompt written to '$cfg_file'\n";
    
    # Run ollama create command
    my $cmd = "ollama create $outfile -f ./$cfg_file";
    print "Running: $cmd\n";
    if (system($cmd) != 0) {
        warn "Failed to run '$cmd': $?\n";
    }
    print "\n";
}

print "Processing complete.\n";
