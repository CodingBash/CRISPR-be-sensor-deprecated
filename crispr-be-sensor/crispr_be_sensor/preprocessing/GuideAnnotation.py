import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from pandarallel import pandarallel
import matplotlib.pyplot as plt
from collections import Counter
import pickle
from datetime import date
from typeguard import typechecked
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
from functools import partial
from itertools import repeat
from typing import List
import re

def save_or_load_pickle(directory, label, py_object = None, date_string = None):
    if date_string == None:
        today = date.today()
        date_string = str(today.year) + ("0" + str(today.month) if today.month < 10 else str(today.month)) + str(today.day)
    
    filename = directory + label + "_" + date_string + '.pickle'
    print(filename)
    if py_object == None:
        with open(filename, 'rb') as handle:
            py_object = pickle.load(handle)
            return py_object
    else:
        with open(filename, 'wb') as handle:
            pickle.dump(py_object, handle, protocol=pickle.HIGHEST_PROTOCOL)     

'''
    Retrieve all pickles with a label, specifically to identify versions available
'''
def display_all_pickle_versions(directory, label):
    return [f for f in listdir(directory) if isfile(join(directory, f)) and label == f[:len(label)]]
    
'''
    Class for processing and storing the annotations of provided genes relevant for the screen
'''
class CodingTilingScreenGeneAnnotations:
    def __init__(self,  gene_symbol_list: List[str], reference_annotation_fn: str="/data/molpath/genomes/hg38/hg38.gtf"):
        self.gene_symbol_list = gene_symbol_list
        self.reference_annotation_fn = reference_annotation_fn
        
        self.annotation_gtf_df = pd.read_table(reference_annotation_fn, header=None)
        # Extract gene_id from annotation, add to new column
        def get_geneid_from_info(info):
            gene_id_start = info.find("gene_id") + len("gene_id") + 2
            gene_id_end = info.find(";")-1
            return info[gene_id_start:gene_id_end]
        self.annotation_gtf_df[9] = self.annotation_gtf_df.iloc[:, 8].apply(get_geneid_from_info)

        # Extract exon_number from annotation, add to new column
        def get_exonnumber_from_info(info):
            exon_num_index = info.find("exon_number")
            if exon_num_index != -1:
                exon_num_start = exon_num_index+len("exon_number")+2
                exon_num_sub = info[exon_num_start:exon_num_start+4]
                exon_num_end = exon_num_sub.find("\"")
                return int(exon_num_sub[:exon_num_end])
        self.annotation_gtf_df[10] = self.annotation_gtf_df.iloc[:, 8].apply(get_exonnumber_from_info)
        
        self.annotation_gtf_gene_df_dict = {gene_symbol:self.annotation_gtf_df[self.annotation_gtf_df.iloc[:, 9]==gene_symbol] for gene_symbol in gene_symbol_list}
        self.gene_coordinates_dict = {gene_symbol: self.__get_sequence_annotations__(annotation_gtf_gene_df) for gene_symbol, annotation_gtf_gene_df in self.annotation_gtf_gene_df_dict.items()}
        
        
        
    def __get_sequence_annotations__(self, gene_annotation_table):
        if gene_annotation_table.iloc[0,6] == "-":
            first_exon = gene_annotation_table[(gene_annotation_table.iloc[:,2] == "exon") & (gene_annotation_table.iloc[:,10] == np.max(gene_annotation_table.iloc[:,10]))]
            start_codon = gene_annotation_table[gene_annotation_table.iloc[:, 2]=="start_codon"]

            last_exon = gene_annotation_table[(gene_annotation_table.iloc[:,2] == "exon") & (gene_annotation_table.iloc[:,10] == np.min(gene_annotation_table.iloc[:,10]))]
            stop_codon = gene_annotation_table[gene_annotation_table.iloc[:, 2]=="stop_codon"]

            utr5_left = start_codon.iloc[0,4]+1
            utr5_right = first_exon.iloc[0,4] + 21 # Adding 21nt buffer for guides just outside border

            utr3_left = last_exon.iloc[0,3] - 21 # Adding 21nt buffer for guides just outside border
            utr3_right = stop_codon.iloc[0,3]-1

            sequence_left = stop_codon.iloc[0,3]
            sequence_right = start_codon.iloc[0,4]

            return {"utr5": (gene_annotation_table.iloc[0,0], utr5_left, utr5_right), "utr3": (gene_annotation_table.iloc[0,0], utr3_left, utr3_right), "sequence": (gene_annotation_table.iloc[0,0], sequence_left, sequence_right)}
        elif gene_annotation_table.iloc[0,6] == "+":
            first_exon = gene_annotation_table[(gene_annotation_table.iloc[:,2] == "exon") & (gene_annotation_table.iloc[:,10] == np.min(gene_annotation_table.iloc[:,10]))]
            start_codon = gene_annotation_table[gene_annotation_table.iloc[:, 2]=="start_codon"]

            last_exon = gene_annotation_table[(gene_annotation_table.iloc[:,2] == "exon") & (gene_annotation_table.iloc[:,10] == np.max(gene_annotation_table.iloc[:,10]))]
            stop_codon = gene_annotation_table[gene_annotation_table.iloc[:, 2]=="stop_codon"]

            utr5_left = first_exon.iloc[0,3] - 21 
            utr5_right =  start_codon.iloc[0,3]-1 # Adding 21nt buffer for guides just outside border

            utr3_left = stop_codon.iloc[0,4]+1
            utr3_right = last_exon.iloc[0,4] + 21 # Adding 21nt buffer for guides just outside border

            sequence_left = start_codon.iloc[0,3]
            sequence_right = stop_codon.iloc[0,4]

            return {"utr5": (gene_annotation_table.iloc[0,0], utr5_left, utr5_right), "utr3": (gene_annotation_table.iloc[0,0], utr3_left, utr3_right), "sequence": (gene_annotation_table.iloc[0,0], sequence_left, sequence_right)}
        else:
            raise Error("Not implemented for strand " + str(gene_annotation_table.iloc[0,6]))
            
    def __repr__(self):
        return str(self.gene_symbol_list) + ";" + str(self.reference_annotation_fn)
    
class GenomeChromosomeSequences:
    def __init__(self, reference_fasta_fn: str, valid_chromosomes: List[str] = None, cores=1):
        self.reference_fasta_fn = reference_fasta_fn
        '''
            Read in reference genome
        '''
        if valid_chromosomes == None:
            valid_chromosomes = ["chr" + str(i) for i in range(1,23)] + ["chrX", "chrY"]
        
        
            
        '''
            TODO 20220723 Unsure if parallelization is working
        '''
        with open(self.reference_fasta_fn,'r') as reference_fasta_file, Pool(cores) as pool:
            self.chromosome_sequences_dict = dict(pool.map(
                GenomeChromosomeSequences.record_elements,
                (record for record in SeqIO.parse(reference_fasta_file, "fasta") if record.id in valid_chromosomes),
                chunksize=1
            ))
    
    @staticmethod
    def record_elements(record): 
        return (record.id, record.seq)
    

class AnnotationsHelperFunctions:
    '''Helper functions for interacting with GTF annotation'''

    @staticmethod
    def get_coord_annotation(guide_chrom, editsite):
        for gene, seq_annotations in screenGeneAnnotations.gene_coordinates_dict.items():
            for element, element_coordinates in seq_annotations.items():
                annotation_id = gene + "_" + element

                element_chrom = element_coordinates[0]
                element_start = element_coordinates[1]
                element_end = element_coordinates[2]

                if (guide_chrom == element_chrom) and (editsite >= element_start) and (editsite <= element_end):
                    return annotation_id

    @staticmethod
    def get_coords(protospacer):
        indices = []
        for chrom in genomeChromosomeSequences.chromosome_sequences_dict:
            coords = [(chrom, "+", match.start(), match.end(), get_coord_annotation(chrom, match.start()+6)) for match in re.finditer(protospacer, str(genomeChromosomeSequences.chromosome_sequences_dict[chrom]))]
            if len(coords) == 0:
                coords = [(chrom, "-", match.start(), match.end(), get_coord_annotation(chrom, match.end()-6)) for match in re.finditer(str(Seq(protospacer).reverse_complement()), str(genomeChromosomeSequences.chromosome_sequences_dict[chrom]))]
                if len(coords) == 0:
                    continue
                else:
                    indices.extend(coords)
            else:
                indices.extend(coords)

        return indices

def annotate_guides(guide_sequences_series, cores=1):
    with Pool(cores) as pool:
        guide_sequences_annotations = pool.map(
            get_coords,
            guide_sequences_series.values)
        
    return guide_sequences_annotations



class CodingTilingScreenGuideSet:
    '''Object containing the guide set'''
    
    guides_sequences_annotations : pd.Series = None
    guide_sequences_is_editable : pd.Series = None
    
    def __init__(self, guide_sequences_fn: str, editing_window_start: int, editing_window_end: int, base: str, sep: str = "\t", header=None, remove_duplicates:bool =True):
        self.editing_window_start = editing_window_start
        self.editing_window_end = editing_window_end
        
        self.guide_sequences_fn = guide_sequences_fn
        
        '''
            Read in guide sequence whitelist
        '''
        self.guide_sequences_series = pd.read_table(self.guide_sequences_fn, sep=sep, header=header).iloc[:, 0]
        if remove_duplicates:
            self.guide_sequences_series = self.guide_sequences_series.apply(lambda guide: guide.upper())
            self.guide_sequences_series = self.guide_sequences_series[~self.guide_sequences_series.duplicated()]
        
        self.guide_sequences_is_editable = self.guide_sequences_series.apply(CodingTilingScreenGuideSet.is_editable, args=(self.editing_window_start, self.editing_window_end, base,))
        
        self.guide_sequences_series.index = self.guide_sequences_series
        self.guide_sequences_is_editable.index = self.guide_sequences_series
        print("{} number of guides".format(self.guide_sequences_series.shape[0]))
    
    @staticmethod
    def is_editable(guide, editing_window_start = 2, editing_window_end = 9, base="A"):
        return base in guide[editing_window_start:editing_window_end].upper()

class GuideAnnotations:
    @staticmethod
    def get_coord_annotation(guide_chrom, editsite):
        for gene, seq_annotations in screenGeneAnnotations.gene_coordinates_dict.items():
            for element, element_coordinates in seq_annotations.items():
                annotation_id = gene + "_" + element

                element_chrom = element_coordinates[0]
                element_start = element_coordinates[1]
                element_end = element_coordinates[2]

                if (guide_chrom == element_chrom) and (editsite >= element_start) and (editsite <= element_end):
                    return annotation_id
    
    @staticmethod
    def get_coords(protospacer):
        indices = []
        for chrom in genomeChromosomeSequences.chromosome_sequences_dict:
            coords = [(chrom, "+", match.start(), match.end(), get_coord_annotation(chrom, match.start()+6)) for match in re.finditer(protospacer, str(genomeChromosomeSequences.chromosome_sequences_dict[chrom]))]
            if len(coords) == 0:
                coords = [(chrom, "-", match.start(), match.end(), get_coord_annotation(chrom, match.end()-6)) for match in re.finditer(str(Seq(protospacer).reverse_complement()), str(genomeChromosomeSequences.chromosome_sequences_dict[chrom]))]
                if len(coords) == 0:
                    continue
                else:
                    indices.extend(coords)
            else:
                indices.extend(coords)

        return indices

    @staticmethod
    def annotate_guides(guide_sequences_series, cores=1):
        with Pool(cores) as pool:
            guide_sequences_annotations = pool.map(
                get_coords,
                guide_sequences_series.values)
            
        return guide_sequences_annotations


if __init__ == "__main__":
    # Load screen gene annotations
    screenGeneAnnotations = CodingTilingScreenGeneAnnotations(gene_symbol_list = ["HBD", "HBA1", "HBA2", "HBE1", "HBB", "HBG1", "HBG2", "DDX6", "BCL11A"])



    # Get reference genome
    ''' Open the cached file of the '''
    print(display_all_pickle_versions(directory="./", label="genomeChromosomeSequences"))

    rerun = False
    if rerun:
        genomeChromosomeSequences = GenomeChromosomeSequences(reference_fasta_fn= '/data/molpath/genomes/hg38/hg38.fa', cores=24)
        save_or_load_pickle(directory="./", label="genomeChromosomeSequences", py_object=genomeChromosomeSequences)
    else:
        genomeChromosomeSequences = save_or_load_pickle(directory="./", label="genomeChromosomeSequences", date_string="2022089")

    

    # Load the cached guide set for the DDX6 screen (with the annotations)
    display_all_pickle_versions(directory="./", label="screenGuideSet")
    rerun = False
    if rerun:
        screenGuideSet = CodingTilingScreenGuideSet(guide_sequences_fn = "../resources/final20mers_unlabeled.csv", editing_window_start = 2, editing_window_end = 9, base = "A")
        guide_sequences_series_annotations = annotate_guides(screenGuideSet.guide_sequences_df, cores=200)
        
        screenGuideSet.guides_sequences_annotations= pd.Series(guide_sequences_series_annotations, index=screenGuideSet.
                                                            guide_sequences_df.index)

        save_or_load_pickle(directory="./", label="screenGuideSet", py_object=screenGuideSet)
    else:
        screenGuideSet = save_or_load_pickle(directory="./", label="screenGuideSet", date_string="20220726")

    # NOTE 20221202 - the screenGuideSet pickle saved before CodingTilingScreenGuideSet included the is_editable field, so adding now. Need to rerun and save a new pickle until this can be removed
    screenGuideSet_tmp = CodingTilingScreenGuideSet(guide_sequences_fn = "../resources/final20mers_unlabeled.csv", editing_window_start = 2, editing_window_end = 9, base = "A")
    screenGuideSet.guide_sequences_is_editable = screenGuideSet_tmp.guide_sequences_is_editable
    screenGuideSet.guide_sequences_series = screenGuideSet_tmp.guide_sequences_series

    # TODO: St
    final20mers_unlabeled_df = pd.read_table("../resources/final20mers_unlabeled.csv", header=None)
    # NOTE: There is also this guide library set based on the nonzero plasmid counts - may be better to use this one
    Steve_MN70_BE_library_plasmid_library_df = pd.read_table("../resources/20220706_plasmid_counts_by_linda/Steve_MN70_BE_library_plasmid_library.txt")

    from matplotlib import pyplot as plt
    screenGuideSet.guides_sequences_annotations.apply(lambda info: len(info)).value_counts().sort_index().plot(kind='bar', figsize=(14,8), title="Number of hg38 genome occurences of protospacers")
    plt.show()

    screenGuideSet.guides_sequences_annotations.apply(lambda row: list(set([index[4] for index in row]))).value_counts().plot(kind='bar', figsize=(14,8), title="Number of multi-annotation protospacer sequences")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Annotation Set")
    plt.ylabel("Guide Count")
    plt.show()

    import upsetplot
    set_counts = screenGuideSet.guides_sequences_annotations.apply(lambda row: list(set([str(index[4]) for index in row]))).value_counts()

    categories = [["non-targeting"] if len(category) == 0 else category for category in list(set_counts.index.values)]
    categories_counts = list(set_counts.values)

    upsetplot.UpSet(upsetplot.from_memberships(categories, data = categories_counts), subset_size="sum")

    guide_sequences_nonannotated = screenGuideSet.guides_sequences_annotations.apply(lambda row: [index for index in row if index[4] == None])
    guide_sequences_nonannotated = guide_sequences_nonannotated[guide_sequences_nonannotated.apply(lambda index: len(index) > 0)]

    guide_sequences_nonannotated.apply(lambda index: index[0][0]).value_counts().plot(kind='bar')
    plt.title("Chromosomes of sequences not within annotated regions")
    plt.show()



    guide_sequences_nonannotated_chr11_coordinates = guide_sequences_nonannotated[guide_sequences_nonannotated.apply(lambda indices: indices[0][0] == "chr11")].apply(lambda indices: indices[0][2])
    guide_sequences_nonannotated_chr11_coordinates[(guide_sequences_nonannotated_chr11_coordinates> 6000000) & (guide_sequences_nonannotated_chr11_coordinates< 60000000)].hist(bins=100)
    plt.show()
    guide_sequences_nonannotated_chr11_coordinates[(guide_sequences_nonannotated_chr11_coordinates<6000000) ].hist(bins=100)
    plt.title("chr11 non-annotated guides")
    plt.xlabel("hg38 chr11 guide start coordinate")
    plt.ylabel("Number of guides per bin")
    plt.show()


        
