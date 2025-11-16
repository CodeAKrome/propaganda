#!/bin/bash

# Configuration
BATCH_SIZE=4  # Number of parallel jobs to run at once
LOG_DIR="./logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Array of commands to run
commands=(
    './batch.sh myanmar "Summarize actions in Myanmar. Focus on any violent actions. Analize actions and motivations."'
    './batch.sh digid "Summarize stories about digital ID. List places and specific actions. Analize actions and motivations."'
    './batch.sh votefraud "Summarize stories having to do with voting fraud in the US, United States. List all details. Analize actions and motivations."'
    './batch.sh genz "Summarize news about Gen-Z protests. List locations and groups involved. Analize actions and motivations."'
    './batch.sh cbdc "Summarize crypto currency, CBDC central bank digital currency, or digital asset legislation and implementation plans. Analize actions and motivations."'
    './batch.sh syriachrist "Summarize events about christians being killed in Syria. Analize actions and motivations."'
    './batch.sh nigeriachrist "Summarize events about christians being killed in Nigeria. Analize actions and motivations."'
    './batch.sh antichrist "Summarize events about christians being killed or otherwise discriminated against. Summarize the locations. Analize actions and motivations."'
    './batch.sh supreme "Summarize current actions by the US Supreme Court. Analize actions and motivations."'
    './batch.sh jobs "Summarize current layoffs and events in the tech market. Analize actions and motivations."'
    './batch.sh antifa "Summarize current actions by ANTIFA. Analize actions and motivations."'
    './batch.sh portland "Summarize actions in Portland Oregon involving ICE, the Mayor and ANTIFA. Analize actions and motivations."'
    './batch.sh chicago "Summarize actions in Chicago. Focus on the Mayor and Governor'\''s actions. Analize actions and motivations."'
    './batch.sh czech "Summarize election and related events in the Czech Republic. Analize actions and motivations."'
    './batch.sh propalestine "Summarize news about pro Palestinian protests. Analize actions and motivations."'
    './batch.sh minerals "Summarize news about critical minerals and rare earths needed for high tech. Analize actions and motivations."'
    './batch.sh russia "Summarize news about Russia. Analize actions and motivations."'
    './batch.sh afghanistan "Summarize news about Afghanistan. Analize actions and motivations."'
    './batch.sh france "Summarize news about France. Analize actions and motivations."'
    './batch.sh netflix "Summarize news about Netflix. Analize actions and motivations."'
    './batch.sh uk "Summarize events in the United Kingdom. Analize actions and motivations."'
    './batch.sh debank "Summarize all actions to debank or deny people bank accounts. Analize actions and motivations."'
    './batch.sh foreigncrim "Summarize all actions involving migrants or illegal immigrants. Analize actions, crimes and motivations."'
    './batch.sh guns "Summarize all stories about US gun laws. Analize actions and motivations."'
    './batch.sh canadaint "Summarize actions taken by the Canadian government to restrict internet access. Analize actions and motivations."'
    './batch.sh canadagun "Summarize actions taken by the Canadian government to restrict guns. Analize actions and motivations."'
    './batch.sh venezuela "Summarize US involvement in Venezuela. Analize actions and motivations."'
    './batch.sh gaza "Summarize events in Gaza. Analize actions and motivations."'
    './batch.sh nepal "Summarize current events in Nepal. Analize actions and motivations."'
    './batch.sh madagascar "Summarize current events in Madagascar. Analize actions and motivations."'
    './batch.sh morocco "Summarize current events in Morocco. Analize actions and motivations."'
    './batch.sh autism "Summarize news linking autism to drug or vaccine causes. Analize actions and motivations."'
    './batch.sh skoreavisa "Summarize news about South Korea and US work visas. Analize actions and motivations."'
    './batch.sh london "Summarize news about London, England. Analize actions and motivations."'
    './batch.sh memphis "Summarize news about Memphis. Analize actions and motivations."'
    './batch.sh kyiv "Summarize news about Kyiv. Analize actions and motivations."'
    './batch.sh disney "Summarize news about Disney. Analize actions and motivations."'
    './batch.sh safrica "Summarize events in South Africa. Analize actions and motivations."'
    './batch.sh japan "Summarize events in Japan. Analize actions and motivations."'
    './batch.sh japanelect "Summarize election events in Japan. Analize actions and motivations."'
    './batch.sh aus "Summarize events in Australia. Analize actions and motivations."'
    './batch.sh germany "Summarize events in Germany, United Kingdom. Analize actions and motivations."'
    './batch.sh typhoon "Summarize any typhoons, their location and damage. Summarize and analize actions taken by authorities."'
    './batch.sh greta "Summarize all actions taken for and against Greta Thunberg and the flotilla. Analize actions and motivations."'
    './batch.sh blackrock "Summarize actions taken by Black Rock. Analize actions, consequences and motivations."'
    './batch.sh vanguard "Summarize actions taken by Vanguard. Analize actions, consequences and motivations."'
    './batch.sh utube "Summarize all legal actions taken involving youtube or their parent company, alphabet, and censorship. Analize actions and motivations."'
    './batch.sh earthquake "Summarize any earthquakes, their location and damage. Summarize and analize actions taken by authorities."'
    './batch.sh govtshut "Summarize details about a US government shutdown. Analize actions and motivations."'
    './batch.sh safarmer "Summarize actions in South Africa taken against White farmers. Analize actions and motivations."'
    './batch.sh farage "Summarize actions taken by Nigel Farage. Analize actions and motivations."'
    './batch.sh japanimm "Summarize actions and protests in Japan about immigration. Analize actions and motivations."'
    './batch.sh hollywood "Summarize actions to tax films made outside the US. Analize actions and motivations."'
    './batch.sh dei "Summarize all actions about DEI, diversity, equity ans inclusion. Analize actions and motivations."'
    './batch.sh hamas "Summarize actions taken by Hamas. Analize actions and motivations."'
    './batch.sh charlie "Summarize events about Charlie Kirk. Analize actions and motivations."'
    './batch.sh moldova "Summarize events about Moldova elections. Analize actions and motivations."'
    './batch.sh ukraine "Summarize actions in the war in Ukraine. Analize actions and motivations."'
    './batch.sh chinasea "Summarize actions in the South China Sea. Analize actions and motivations."'
    './batch.sh nycrace "Summarize actions in race for New York mayor. Analize actions and motivations."'
    './batch.sh houthis "Summarize actions taken by the Houthis. Analize actions and motivations."'
    './batch.sh india "Summarize current events in India. Analize actions and motivations."'
    './batch.sh iran "Summarize current events taken by Iran. Analize actions and motivations."'
    './batch.sh netanyahu "Summarize current actions by Netanyahu. Analize actions and motivations."'
    './batch.sh skorea "Summarize current actions in South Korea. Analize actions and motivations."'
    './batch.sh soybean "Summarize current situation with US soybean farmers. Analize actions and motivations."'
)

# Function to run a single command with logging
run_command() {
    local cmd="$1"
    local topic=$(echo "$cmd" | sed -n 's/.*\.\/batch\.sh \([^ ]*\).*/\1/p')
    local logfile="$LOG_DIR/${topic}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $topic"
    eval "$cmd" > "$logfile" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $topic"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed: $topic (exit code: $exit_code)"
    fi
    
    return $exit_code
}

export -f run_command
export LOG_DIR

# Main execution
echo "=========================================="
echo "Starting parallel batch processing"
echo "Batch size: $BATCH_SIZE jobs"
echo "Total commands: ${#commands[@]}"
echo "Log directory: $LOG_DIR"
echo "=========================================="
echo ""

# Run commands in parallel using GNU parallel or xargs
if command -v parallel &> /dev/null; then
    # Use GNU parallel if available (preferred)
    printf '%s\n' "${commands[@]}" | parallel -j "$BATCH_SIZE" --line-buffer run_command {}
else
    # Fallback to xargs with manual control
    echo "GNU parallel not found, using xargs as fallback"
    printf '%s\n' "${commands[@]}" | xargs -I {} -P "$BATCH_SIZE" bash -c 'run_command "$@"' _ {}
fi

echo ""
echo "=========================================="
echo "All batch jobs completed!"
echo "Check logs in: $LOG_DIR"
echo "=========================================="
