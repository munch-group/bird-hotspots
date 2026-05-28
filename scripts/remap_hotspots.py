import numpy as np
import pandas as pd
import bisect
import sys

np.random.seed(7)

import genominterv

def get_remapped(query, annot):
    "Remaps all annot that does not overlap query"
    
    remapped = (genominterv.remap_interval_data(query, annot)
                      # exclude windows between ends of chromosomes and a hotspot:
                      .loc[lambda df: ~df.start_prox.isnull() & ~df.end_prox.isnull()]
                     )
    return remapped

def get_overlapping(query, annot):
    "Gets all query that spans an annot"
    
    query_grouped = query.groupby('chrom')
    annot_grouped = annot.groupby('chrom')

    df_list = list()
    for chrom in chromosomes:
        query_group = query_grouped.get_group(chrom)
        annot_group = annot_grouped.get_group(chrom)

        query_starts = query_group.start.tolist()
        query_ends = query_group.end.tolist()

        # loop over hotspot centers
        for hotspot_center in annot_group.pos:
            # find index of overlapping hotspot, if any
            i = bisect.bisect_right(query_starts, hotspot_center)
            if i and hotspot_center < query_ends[i-1]:
                idx = i-1
                row_df = query_group.iloc[idx:idx+1] # slice to get row as data frame
                df_list.append(row_df.assign(hotspot_center=hotspot_center))

    query_overlap = pd.concat(df_list).reset_index(drop=True)
    
    # Add the extra columns so concatention is possible and set `start` and `end` to 0:
    query_overlap['start_prox'] = query_overlap.hotspot_center
    query_overlap['end_prox'] = query_overlap.hotspot_center
    query_overlap.drop(columns=['hotspot_center'], inplace=True)
    
    query_overlap['start_orig'] = query_overlap.start
    query_overlap['end_orig'] = query_overlap.end
    query_overlap['start'] = 0
    query_overlap['end'] = 0

    return query_overlap


def remap_data(query, annot):
    "Remap hotspot centers relative to other annotation intervals"
    
    annot_centers = annot.assign(start=annot.pos, end=annot.pos)
    
    query_dist = get_remapped(query, annot_centers)

    # must be single base annotation annotation
    # for this to work (hotspot centers)
    assert not (annot_centers.end - annot_centers.start).sum()    
    query_overlap = get_overlapping(query, annot_centers)

    merged = pd.concat([query_overlap, query_dist], sort=True)
    
    merged['pos'] = (merged.start + (merged.end - merged.start) / 2).astype(np.double)

    merged['bin'] = merged.pos.round(-3) # round to closest thousand

    # A bit of a hack: we want there to be one wy
    # indow bin for each hotspot.
    # In the special case (there is only one) where pos is 500 the bin is 
    # rounded down to 0, leaving two zero bins and no 1000kb bin. 
    # So we just set that to 1000 (effectively rounding that 500 up insetead of down.):
    merged.loc[merged.pos == 500.0, 'bin'] = 1000
    merged.loc[merged.pos == -500.0, 'bin'] = -1000
    
    return merged


def optimize_data_frame(df, inplace=False, down_int='integer'):
    # down_int can be 'unsigned'
    
    if inplace:
        converted_df = df
    else:
        converted_df = pd.DataFrame()

    floats_optim = (df
                    .select_dtypes(include=['float'])
                    .apply(pd.to_numeric,downcast='float')
                   )
    converted_df[floats_optim.columns] = floats_optim

    ints_optim = (df
                    .select_dtypes(include=['int'])
                    .apply(pd.to_numeric,downcast=down_int)
                   )
    converted_df[ints_optim.columns] = ints_optim

    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            converted_df[col] = df[col].astype('category')
        else:
            converted_df[col] = df[col]

    unchanged_cols = df.columns[~df.columns.isin(converted_df.columns)]
    converted_df[unchanged_cols] = df[unchanged_cols]

    # keep columns order
    converted_df = converted_df[df.columns]      
            
    return converted_df


_, hotspot_data_file, cgi_data_file, promoter_data_file, tss_data_file, tes_data_file, \
    outfile_hotspots_rel_cgi, outfile_hotspots_rel_promoter, outfile_hotspots_rel_tss, outfile_hotspots_rel_tes = sys.argv

chromosomes = ['1', '1A', '2', '3', '4', '4A', '5', '6', '7', 
               '8', '9', '10', '11', '12', '13', '14', '15']

# read hotspot data
hotspots = pd.read_csv(hotspot_data_file, names=['chrom', 'start', 'end'], sep='\t')
hotspots['pos'] = (hotspots.start + (hotspots.end - hotspots.start) / 2).round().astype(int)
hotspots['chrom'] = hotspots.chrom.str.replace('chr', '')
hotspots.rename(columns={'stop': 'end'}, inplace=True)

# read cgi data
cgi = pd.read_csv(cgi_data_file, sep='\t')
cgi['pos'] = (cgi.start + (cgi.end - cgi.start) / 2).round().astype(int)
cgi['chrom'] = cgi['chr'].str.replace('chr', '')

# read promoter data
promoters = pd.read_csv(promoter_data_file, names=['chrom', 'start', 'end'], sep='\t')
promoters['pos'] = (promoters.start + (promoters.end - promoters.start) / 2).round().astype(int)
promoters['chrom'] = promoters.chrom.str.replace('chr', '')

# read tss data
tss = pd.read_csv(tss_data_file, names=['chrom', 'start', 'end'], sep='\t')
tss['pos'] = (tss.start + (tss.end - tss.start) / 2).round().astype(int)
tss['chrom'] = tss.chrom.str.replace('chr', '')

# read tes data
tes = pd.read_csv(tes_data_file, names=['chrom', 'start', 'end'], sep='\t')
tes['pos'] = (tes.start + (tes.end - tes.start) / 2).round().astype(int)
tes['chrom'] = tes.chrom.str.replace('chr', '')

# remap hotspots relative to cgi and write file
remapped = remap_data(hotspots, cgi)
remapped.to_csv(outfile_hotspots_rel_cgi, sep='\t', index=False)

# remap hotspots relative to tss and write file
remapped = remap_data(hotspots, tss)
remapped.to_csv(outfile_hotspots_rel_tss, sep='\t', index=False)

# remap hotspots relative to tes and write file
remapped = remap_data(hotspots, tes)
remapped.to_csv(outfile_hotspots_rel_tes, sep='\t', index=False)

# remap hotspots relative to tes and write file
remapped = remap_data(hotspots, promoters)
remapped.to_csv(outfile_hotspots_rel_promoter, sep='\t', index=False)

#python scripts/test.py /project/Birds/faststorage/data/composition_cpg/TYTAL /project/Birds/faststorage/data/bed/hotspots.bed /project/Birds/faststorage/data/bed/CGI-taeGut1.txt /project/Birds/faststorage/data/composition_cpg_remapped/TYTAL.hotspot_rel_cgi.txt 