/**
 * Test the new parsing logic for raw_output
 */

const rawOutput = '"dir":"L":0.2,"C":0.3,"R":0.5,"deg":"L":0.1,"M":0.4,"H":0.5,"reason":"The article frames the radical left-wing agenda as destroying our country."';

const result = {
  dir: { L: 0, C: 0, R: 0 },
  deg: { L: 0, M: 0, H: 0 },
  reason: ''
};

console.log('Parsing raw_output:', rawOutput);

// Find all key:value pairs (with or without quotes on key)
const allPairs = [];

// Match quoted keys like "L":0.3
const quotedRegex = /"(L|C|R|M|H)"\s*:\s*([\d.]+)/g;
let match;
while ((match = quotedRegex.exec(rawOutput)) !== null) {
  allPairs.push({ key: match[1], value: parseFloat(match[2]) });
}

// Match unquoted keys like M:0.4 (but not inside strings)
const unquotedRegex = /[,}]?\s*(M|H)\s*:\s*([\d.]+)/g;
while ((match = unquotedRegex.exec(rawOutput)) !== null) {
  // Check if this M or H is already captured
  const alreadyCaptured = allPairs.some(p => p.key === match[1]);
  if (!alreadyCaptured) {
    allPairs.push({ key: match[1], value: parseFloat(match[2]) });
  }
}

console.log('All pairs found:', allPairs);

// Separate direction and degree values
const lcrValues = allPairs.filter(p => ['L', 'C', 'R'].includes(p.key));
const mhValues = allPairs.filter(p => ['M', 'H'].includes(p.key));

console.log('L/C/R values:', lcrValues);
console.log('M/H values:', mhValues);

// First 3 L/C/R values are direction
const dirValues = lcrValues.slice(0, 3);
// Remaining L value (if any) is degree L
const degL = lcrValues.length > 3 ? lcrValues[3] : null;

console.log('Direction values:', dirValues);
console.log('Degree L:', degL);

// Assign direction values
dirValues.forEach(v => {
  result.dir[v.key] = v.value;
});

// Assign degree values
if (degL) result.deg.L = degL.value;
mhValues.forEach(v => {
  result.deg[v.key] = v.value;
});

// Extract reason
const reasonMatch = rawOutput.match(/"reason"\s*:\s*"([^"]+)"/);
if (reasonMatch) result.reason = reasonMatch[1];

console.log('Parsed result:', JSON.stringify(result, null, 2));