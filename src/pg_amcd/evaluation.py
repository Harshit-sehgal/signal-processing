import os
import re
import glob
import openpyxl
from typing import List, Dict, Any

def build_dataset_index(
    input_dir: str, 
    metadata_excel_path: str
) -> List[Dict[str, Any]]:
    """Scans all MAT files in input_dir and maps them to metadata rows in the Excel combinations workbook.
    
    Returns:
        A list of dictionaries representing the master index table.
        
    Raises:
        ValueError: If there are duplicate recordings or missing machining parameters.
    """
    if not os.path.exists(metadata_excel_path):
        raise FileNotFoundError(f"Metadata Excel file not found: {metadata_excel_path}")
        
    wb = openpyxl.load_workbook(metadata_excel_path)
    
    # 1. Parse all excel sheets to build combinations master list
    combinations = []
    sheets_to_load = [s for s in wb.sheetnames if s not in ['recording times']]
    
    for sname in sheets_to_load:
        ws = wb[sname]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
            
        header = [str(c).strip().lower() if c is not None else '' for c in rows[0]]
        
        # Column mappings
        idx_ld = -1
        for name in ['l/d', 'ld']:
            if name in header:
                idx_ld = header.index(name)
                break
                
        idx_rpm = -1
        for name in ['rpm', 'spindle speed']:
            if name in header:
                idx_rpm = header.index(name)
                break
                
        idx_doc = -1
        for name in ['doc (in)', 'corrected doc (in)', 'doc']:
            if name in header:
                idx_doc = header.index(name)
                break
                
        idx_state = -1
        for name in ['state', 'chatter/nochatter', 'status']:
            if name in header:
                idx_state = header.index(name)
                break
                
        idx_feed = -1
        for name in ['feed (in/rev)', 'feed', 'feedrate']:
            if name in header:
                idx_feed = header.index(name)
                break
                
        if idx_rpm == -1 or idx_doc == -1 or idx_state == -1:
            continue
            
        for r in rows[1:]:
            if r[idx_rpm] is not None and r[idx_doc] is not None:
                # Resolve L/D
                if idx_ld != -1 and r[idx_ld] is not None:
                    try:
                        ld = float(r[idx_ld])
                    except ValueError:
                        ld = 2.0
                else:
                    m = re.search(r'rod(\d+)p?(\d*)inch', sname)
                    if m:
                        ld = float(m.group(1)) + (float(m.group(2))/10.0 if m.group(2) else 0.0)
                    else:
                        ld = 3.5 if '3p5' in sname else 4.125 if '4p125' in sname else 4.5 if '4p5' in sname else 2.0
                        
                try:
                    rpm_val = int(r[idx_rpm])
                    doc_val = float(r[idx_doc])
                except (ValueError, TypeError):
                    continue
                    
                state_val = str(r[idx_state]).strip().lower() if r[idx_state] is not None else ''
                feed_val = float(r[idx_feed]) if (idx_feed != -1 and r[idx_feed] is not None) else 0.002
                
                # Standardize state label
                if 'no chatter' in state_val or 'nochatter' in state_val:
                    label = 'stable'
                elif 'mild' in state_val or 'incipient' in state_val or 'intermittent' in state_val:
                    label = 'incipient'
                else:
                    label = 'chatter'
                    
                combinations.append({
                    'ld': ld,
                    'rpm': rpm_val,
                    'doc': doc_val,
                    'feed': feed_val,
                    'label': label
                })

    # 2. Scan MAT files and map them
    ld_map = {
        '2inch_stickout': 2.0, 
        '2p5inch_stickout': 2.5, 
        '3p5inch_stickout': 3.5, 
        '4p5inch_stickout': 4.5
    }
    
    mat_files = glob.glob(os.path.join(input_dir, "**/*.mat"), recursive=True)
    mat_files = [f for f in mat_files if not f.endswith("combinations.xlsx") and "~lock" not in f]
    
    index_table = []
    seen_recording_ids = set()
    
    for f in mat_files:
        folder = os.path.basename(os.path.dirname(f))
        base = os.path.basename(f)
        base_no_ext = base.replace(".mat", "")
        
        ld = ld_map.get(folder)
        if ld is None:
            raise ValueError(f"Unknown stickout folder name: {folder} for file {f}")
            
        parts = base_no_ext.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid filename structure for file {f}")
            
        prefix = parts[0]
        try:
            rpm = int(parts[1])
            doc_str = re.sub('[^0-9]', '', parts[2])
            doc = float(doc_str) / 1000.0
        except ValueError:
            raise ValueError(f"Failed to parse RPM or DOC from filename {base}")
            
        # Parse run number if present, e.g. u_570_015_3 -> run 3
        run_id = "1"
        if len(parts) >= 4:
            run_id = parts[3]
            
        recording_id = f"ld_{ld}_rpm_{rpm}_doc_{doc}_state_{prefix}_run_{run_id}"
        if recording_id in seen_recording_ids:
            raise ValueError(f"Duplicate recording ID detected: {recording_id}")
        seen_recording_ids.add(recording_id)
        
        # Match combination from Excel
        match = None
        for row in combinations:
            if abs(row['ld'] - ld) < 0.1 and row['rpm'] == rpm and abs(row['doc'] - doc) < 0.001:
                match = row
                break
                
        # Label resolution (with fallback to filename prefix)
        if match:
            label = match['label']
            feed = match['feed']
        else:
            # Fallback to prefix
            feed = 0.002
            if prefix == 's':
                label = 'stable'
            elif prefix == 'i':
                label = 'incipient'
            else:
                label = 'chatter'
                
        index_table.append({
            'recording_id': recording_id,
            'file_path': f,
            'stickout': ld,
            'rpm': rpm,
            'depth_of_cut': doc,
            'feed_rate': feed,
            'tooth_count': 1, # default single tooth cutter
            'tool_id': "tool_01",
            'machine_id': "lathe_01",
            'sensor_id': "accelerometer_01",
            'label': label,
            'chatter_onset_time': 0.0,
            'experiment_run_id': run_id
        })
        
    return index_table
