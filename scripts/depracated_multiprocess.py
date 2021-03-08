    # def run(self, test=False):
    #     t_start = time.time()
    #     src_batch_dict_list = []
    #     tgt_batch_dict_list = []
        
    #     # num_splits = num_intervals - 1 e.g. -|-|- 
    #     num_pools = multiprocessing.cpu_count() if not test else 2
    #     pool = multiprocessing.Pool(num_pools) 
    #     num_splits = (multiprocessing.cpu_count() - 2) // 2  
    #     src_batch_iter = self.interval_split(num_splits, len(self.src_tsv))
    #     tgt_batch_iter = self.interval_split(num_splits, len(self.tgt_tsv))
        
    #     count = 0
    #     for src_batch_inds in src_batch_iter:
    #         count += 1
    #         src_batch_dict_list.append(pool.apply_async(self.batch_mappings, args=(src_batch_inds, False, )))
    #         # test round for one batch
    #         if test == True:
    #             break 
    #     # assert count == num_splits + 1
        
    #     count = 0
    #     for tgt_batch_inds in tgt_batch_iter:
    #         count += 1
    #         tgt_batch_dict_list.append(pool.apply_async(self.batch_mappings, args=(tgt_batch_inds, True, )))
    #         if test == True:
    #             break 
    #     # assert count == num_splits + 1
        
    #     pool.close()
    #     pool.join()
        
    #     for result in src_batch_dict_list:
    #         for k, v in result.get().items():
    #             self.src2tgt_mappings_tsv.iloc[k] = v   
    #     for result in tgt_batch_dict_list:
    #         for k, v in result.get().items():
    #             self.tgt2src_mappings_tsv.iloc[k] = v   
        
    #     t_end = time.time()
    #     t = t_end-t_start
    #     self.log_print('the program time is :%s' %t)