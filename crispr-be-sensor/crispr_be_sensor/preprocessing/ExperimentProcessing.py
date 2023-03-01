class CodingTilingScreenSampleInformation:
    def __init__(self, sample_df: pd.DataFrame, plasmid_df: pd.DataFrame, screenGuideSet: CodingTilingScreenGuideSet):
        self.sample_df = sample_df
        self.plasmid_df = plasmid_df
        self.screenGuideSet = screenGuideSet
    
    def compute_sample_guide_counts(self, hamming_threshold_strict: int = 5, hamming_threshold_dynamic: bool = False, cores: int = 1):
        # Validate that all the filenames exist before inferencing:
        for fn in np.concatenate([self.sample_df["filename"], self.plasmid_df["filename"]]):
            if os.path.exists(fn) != True:
                raise Exception("One or more files do not exist: {}".format(fn))
        print("All files exist")
        self.sample_count_results = self.sample_df["filename"].apply(lambda filename: get_guide_counts(screenGuideSet.guide_sequences_series, filename, hamming_threshold_strict=hamming_threshold_strict, hamming_threshold_dynamic=hamming_threshold_dynamic, cores=cores))
        self.plasmid_count_results = self.plasmid_df["filename"].apply(lambda filename: get_guide_counts(screenGuideSet.guide_sequences_series, filename, hamming_threshold_strict=hamming_threshold_strict, hamming_threshold_dynamic=hamming_threshold_dynamic, cores=cores))
        
        