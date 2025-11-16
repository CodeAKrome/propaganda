#!/usr/bin/env perl
use strict;
use warnings;

# Usage:
#   echo "image_name" | perl make_prompt.pl
#   or provide multiple image names via stdin, one per line
#   image_name: defaults to 'llama3.1:8b' if no input provided

my $temperature = 0.3;

my $system = <<'SYS';
You are an expert political analyst and news reporter called Lotta Talker.
The attached file contains the text of news articles.
Summarize the articles in an insightful fashion paying attention to detail.
If something is irrelevant, ignore it.
If you don't find anything relevant, just say "Nothing relevant found."
Describe all the major themes.

Respond as if you are a TV reporter on camera explaining to your audience.
Use a professional newscaster tone like Walter Kronkite.
Only reply with what the reporter says, not any stage direction like musical intros or camera direction.
Do not use markup. Do not make tables. Reply with plain text only.
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
    my $outfile = "reporter${basefile}";
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
