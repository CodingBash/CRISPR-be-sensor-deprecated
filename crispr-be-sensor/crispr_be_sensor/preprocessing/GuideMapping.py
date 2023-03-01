from multiprocessing import Pool
from collections import Counter
import gzip
import random
from enum import Enum
from typing import Callable
from typing import Union, List, Mapping, Tuple, Optional


class GuideHammingMapping:
    '''
        Vector encoding of each base
    '''
    non_ambigious_encoding_dict = dict({
        "A": np.asarray([1,0,0,0]),
        "C": np.asarray([0,1,0,0]),
        "G": np.asarray([0,0,1,0]),
        "T": np.asarray([0,0,0,1]),
        "U": np.asarray([0,0,0,1])
    })

    @staticmethod
    def ambiguity_encoder(bases):
        '''Helper function for retrieving the encoding for ambigious bases based on IUPAC codes'''
        return np.logical_or.reduce([non_ambigious_encoding_dict[base] for base in bases]).astype(int)

    ''' Final dictionary for getting encoding of each IUPAC base '''
    full_encoding_dict = dict({
        "A": non_ambigious_encoding_dict["A"],
        "C": non_ambigious_encoding_dict["C"],
        "G": non_ambigious_encoding_dict["G"],
        "T": non_ambigious_encoding_dict["T"], 
        "R": ambiguity_encoder(["A", "G"]),
        "Y": ambiguity_encoder(["C", "T"]),
        "S": ambiguity_encoder(["G", "C"]),
        "W": ambiguity_encoder(["A", "T"]),
        "K": ambiguity_encoder(["G", "T"]),
        "M": ambiguity_encoder(["A", "C"]),
        "B": ambiguity_encoder(["C", "G", "T"]),
        "D": ambiguity_encoder(["A", "G", "T"]),
        "H": ambiguity_encoder(["A", "C", "T"]),
        "V": ambiguity_encoder(["A", "C", "G"]),
        "N": ambiguity_encoder(["A", "C", "G", "T"]),
    })


    @staticmethod
    def encode_DNA_base(char):
        '''Main function to encode a single base'''
        return full_encoding_dict[char]
    # Vectorized function for a string (i.e. gRNA)
    encode_DNA_base_vectorized = staticmethod(np.vectorize(GuideMapping.encode_DNA_base, signature='()->(n)')) 

    def numpify_string(string):
        '''Function for converting string (i.e. gRNA) into a np array of chars  - may be deprecated (NOTE 20221202)'''
        return np.array(list(string), dtype=str)
    
    numpify_string_vectorized = staticmethod(np.vectorize(GuideMapping.numpify_string, signature='()->(n)')) # Vectorize the function

    @staticmethod
    def encode_guide_series(guide_series):
        guide_numpy = guide_series.to_numpy(dtype=object)
        guide_numpy = guide_numpy.astype(str)
        guide_numpy_char = np.array(list(map(list, guide_numpy))) # Map into a list of list of characters
        guide_numpy_encoding = GuideMapping.encode_DNA_base_vectorized(guide_numpy_char)
        return guide_numpy_encoding

    @staticmethod
    def retrieve_unmatched_bases(obs_encoded, whitelist_guide):
        '''
            Utility function - not really used
        '''
        return ((obs_encoded*true_encoded).sum(axis=1)^1)

    @staticmethod
    def retrieve_hamming_distance_whitelist(target_guide_encoded, whitelist_guide_encoded):
        '''
            This takes a encoded guide sequence and a list of encoded whitelisted guides and matrix computes the hamming distance of the 
            encoded guide across all whitelisted guides in a single operation
            
            (target_guide_encoded*whitelist_guide_encoded[:, np.newaxis]).sum(axis=3) # Determines 
            
        '''
        return ((target_guide_encoded*whitelist_guide_encoded[:, np.newaxis]).sum(axis=3)^1).sum(axis=2).flatten()


class GuideParsingStrategies:
    @typechecked
    def parse_read_positional(read_sequence: Union[str, Seq], position_start: int, position_end: int) -> Union[str, Seq]:  
        return read_sequence[position_start:position_end]

    @typechecked
    def parse_read_left_flank(read_sequence: Union[str, Seq], left_flank:Union[str, Seq], guide_sequence_length:int) -> Union[str, Seq]: 
        position_start = read_sequence.find(left_flank) + len(left_flank)
        return read_sequence[position_start:position_start+guide_sequence_length]

    @typechecked
    def parse_read_right_flank(read_sequence: Union[str, Seq], right_flank:Union[str, Seq], guide_sequence_length:int) -> Union[str, Seq]:
        position_end = read_sequence.find(right_flank);
        return read_sequence[position_start:position_start+guide_sequence_length]


@typechecked
def parse_guide_sequence(read_sequence: Union[str, Seq], parser_function: Callable) -> Union[str, Seq]:
        '''Extract the guide sequence from the read provided'''
        read_guide_sequence = parser_function(read_sequence)
        return read_guide_sequence


class GuideCountError(Enum):
    NO_MATCH = "No guide found within hamming distance"
    MULTIPLE_MATCH = "Multiple exact matches found for guide (likely a truncated guide read assuming guide series is unique)"
    NO_GUIDE_WITH_SAME_LENGTH = "No whitelisted guides with same length as observed guide - maybe try enabling truncating whitelisted guides"

class GuideMapping:
    parse_guide_sequence = staticmethod(parse_guide_sequence)
    

    @typechecked
    def retrieve_fastq_guide_sequences(guide_sequences_series: Union[List[str], pd.Series], fastq_file: str, cores: int=1) -> Union[List[str], List[Seq]]:
        '''Iterate over all the reads in the FASTQ (parallelized) and retrieve the observed guide sequence'''
        parse_read_left_flank_p = partial(parse_read_left_flank, left_flank="CACCG", guide_sequence_length=20)
        parse_guide_sequence_p = partial(parse_guide_sequence, parser_function=parse_read_left_flank_p)
        
        with gzip.open(fastq_file, "rt", encoding="utf-8") as handle, Pool(cores) as pool:
            fastq_guide_sequences = pool.map(
            parse_guide_sequence_p,
            (seq.seq for seq in SeqIO.parse(handle, 'fastq')),
            chunksize=2000,
            )

        return fastq_guide_sequences

    '''
        Given an observed, potentially self-edited guide (in 'row["observed_sequence"]'), try and find the true guide sequence from the whitelist ("guide_sequences_series") based on hamming distance
    '''
    @typechecked
    def infer_true_guides(row: pd.Series, guide_sequences_series: Union[List[str], pd.Series], consider_truncated_sequences: bool = True, hamming_threshold: int = 3):
        observed_guide_sequence = str(row["observed_sequence"])
        
        # If considering truncated guides, truncate all whitelist guides to the same size of the observed guide, else only consider whitelisted guides of the same length (likely all guides provided are same length of 20nt)
        if consider_truncated_sequences == True:
            guide_sequences_series = guide_sequences_series.apply(lambda guide: guide[0:len(observed_guide_sequence)])
        else:
            guide_sequences_series = guide_sequences_series[guide_sequences_series.str.len() == len(observed_guide_sequence)]
            if len(guide_sequences_series) == 0:
                return GuideCountError.NO_GUIDE_WITH_SAME_LENGTH 
            
        # Determine if there are exact matches, hopefully just a single match
        guide_sequences_series_match = guide_sequences_series[guide_sequences_series == observed_guide_sequence]
        
        # If there is a single exact match, great, no need for fancy mat
        if len(guide_sequences_series_match) == 1: # Exact match found, return
            return guide_sequences_series_match.index[0]
        
        # If no matches, possible a self-edit or a sequencing error, try and find guide with lowest hamming distance
        elif len(guide_sequences_series_match) == 0: # No match found, search based on hamming distance
            
            # Encode the whitelisted guides 
            # NOTE 20221202: Potentially improve efficiency by passing in the encoded guide series (assuming no truncation) so that this does not have to be ran on every guide
            guide_sequences_series_encoded = encode_guide_series(guide_sequences_series)
            
            # Encode the observed guide
            observed_guide_sequence_encoded = encode_DNA_base_vectorized(numpify_string_vectorized(observed_guide_sequence)) 
            
            # Calculate the hamming distance of the guide with all whitelisted guides
            observed_guide_sequence_dists = retrieve_hamming_distance_whitelist(observed_guide_sequence_encoded, guide_sequences_series_encoded)
            
            # Get the minimum hamming distance calculated
            hamming_min = observed_guide_sequence_dists.min()
            
            # Get all whitelisted guides with the minimum hamming distance (could be multiple)
            guides_with_hamming_min = guide_sequences_series[np.where(observed_guide_sequence_dists == hamming_min)[0]]
            
            # If the minimum hamming distance is greater than the specified threshold, then the guide is too ambigious, so no match.
            if hamming_min >= hamming_threshold:
                return GuideCountError.NO_MATCH
            
            # If there are multiple guides with the minimum hamming distance, then the guide is ambigious, so no mapping (due to multiple match)
            elif len(guides_with_hamming_min) > 1:
                return GuideCountError.MULTIPLE_MATCH
            
            # Else if there is 1 guide with the match, then return the match
            else:
                return guides_with_hamming_min.index[0]
        
        # Else if there are multiple exact match, which should never occur unless the whitelisted guide list is not unique, then return multiple match.
        else:
            return GuideCountError.MULTIPLE_MATCH
            #raise Exception("Multiple exact matches of the provided whitelisted guides - there are likely duplicates in the provided whitelist, please remove. Observed guide={}, guide matches={}".format(observed_guide_sequence, guide_sequences_series_match)) # NOTE 12/6/22: REMOVED THIS EXCEPTION - another reason is from truncated guides having multiple matches. In production code, just make sure to ensure that the whitelist is the set.

    '''
        This performs simulation of determining how many mutations it takes for a guide to be ambigiously mapped based on hamming distance.
        
        This is useful in determing the ideal hamming distance threshold specific to a guide library
    '''
    @typechecked
    def determine_hamming_threshold(guide_sequences_series: Union[List[str],pd.Series], sample_count: int = 100, quantile: float = 0.05):
        guide_sequences_series_encoded = encode_guide_series(guide_sequences_series)
        
        mutation_count_until_nonunique = []
        for i in range(sample_count):
            # Sample a guide from whitelist
            sampled_guide = guide_sequences_series.sample()[0]
            
            # Generate position orders to "mutate
            guide_position_list= list(range(len(sampled_guide)))
            random.shuffle(guide_position_list)
            
            # Create temporary variable to represent mutated guide
            current_guide_sequence = sampled_guide
            
            # Iterate through positions to mutate
            for iteration, position in enumerate(guide_position_list):
                
                # Mutate the guide
                current_guide_sequence_separated = list(current_guide_sequence)
                guide_position_nt = current_guide_sequence_separated[position]
                nt_list = ["A", "C", "T", "G"]
                nt_list.remove(guide_position_nt.upper())
                new_nt = random.sample(nt_list, 1)[0]
                current_guide_sequence_separated[position] = new_nt
                current_guide_sequence = "".join(current_guide_sequence_separated)
                current_guide_sequence_encoded = encode_DNA_base_vectorized(numpify_string_vectorized(current_guide_sequence)) 
                
                hamming_distances =  retrieve_hamming_distance_whitelist(current_guide_sequence_encoded, guide_sequences_series_encoded)
                if len(np.where(hamming_distances == hamming_distances.min())[0]) > 1:
                    mutation_count_until_nonunique.append(iteration+1)
                    break   
        mutation_count_until_nonunique = pd.Series(mutation_count_until_nonunique)
        return mutation_count_until_nonunique.quantile(quantile)
                                
    '''
        Take in input FASTQ filename, and a set of whitelisted guide sequences
    '''
    @typechecked
    def get_guide_counts(guide_sequences_series: pd.Series, fastq_fn: str, hamming_threshold_strict: int = 3, hamming_threshold_dynamic: bool = False, cores: int=1):
        # Retrieve all observed guide sequences
        print("Retrieving FASTQ guide sequences and counting: " + fastq_fn)
        observed_guide_sequences = retrieve_fastq_guide_sequences(guide_sequences_series, fastq_fn, cores=cores)
        
        # Count each unique observed guide sequence
        observed_guide_sequences_counts = Counter(observed_guide_sequences)
        
        # Convert observed
        observed_guides_df = pd.DataFrame({"observed_sequence":[str(sequence) for sequence in observed_guide_sequences_counts.keys()], "observed_counts":observed_guide_sequences_counts.values()})
    
        # Set the hamming distance
        if hamming_threshold_dynamic:
            hamming_threshold = int(determine_hamming_threshold(guide_sequences_series, sample_count = 100, quantile = 0.05))
            print("Hamming threshold is " + str(hamming_threshold))
        else:
            hamming_threshold = int(hamming_threshold_strict)
            
        # Infer whitelist guides from observed guides
        print("Inferring the true guides from observed guides")
        pandarallel.initialize(progress_bar=True, nb_workers=cores, use_memory_fs=False)
        inferred_true_guides_df = observed_guides_df.parallel_apply(infer_true_guides, args=(guide_sequences_series,True, hamming_threshold, ), axis=1)
        observed_guides_df["inferred_guides"] = inferred_true_guides_df
        

        # Guides passed
        guide_sequences_passed = observed_guides_df[observed_guides_df["inferred_guides"].apply(lambda guide : type(guide) != GuideCountError)]

        # QC: Calculate number of guides that were unassigned
        guide_sequences_unassigned_counts = observed_guides_df[observed_guides_df["inferred_guides"].apply(lambda guide : guide == GuideCountError.NO_MATCH)]
        # QC: Calculate number of guides with multiple inferred guides
        guide_sequences_multiple_counts = observed_guides_df[observed_guides_df["inferred_guides"].apply(lambda guide : guide == GuideCountError.MULTIPLE_MATCH)]
        # QC: Calculate percent mapped
        percent_mapped = guide_sequences_passed["observed_counts"].sum()/observed_guides_df["observed_counts"].sum()
        
        # Retrieve the observed sequences that were mapped and set the inferred guides
        true_guide_sequence_counter = Counter()
        for i in range(guide_sequences_passed.shape[0]):
            true_guide_sequence_counter[guide_sequences_passed.iloc[i, 2]] += guide_sequences_passed.iloc[i, 1]
        
        guide_sequences_series_counts = guide_sequences_series.apply(lambda guide: true_guide_sequence_counter[guide])
        guide_sequences_series_counts.index = guide_sequences_series
        
        qc_dict = {"guide_sequences_unassigned_counts":guide_sequences_unassigned_counts.sum(), "guide_sequences_multiple_counts": guide_sequences_multiple_counts.sum(), "total_guide_counts": observed_guides_df["observed_counts"].sum(), "percent_mapped": percent_mapped}
        return observed_guide_sequences, guide_sequences_series_counts, observed_guides_df, qc_dict

