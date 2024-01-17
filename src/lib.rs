
pub mod histogram;
pub mod bitstream;



struct FseTable {
    tableLog: usize,
    table: Vec<u16>,
}

impl FseTable {
    fn new(hist: &[i16; 256], log2_size: u32) -> Self {
        assert!(log2_size >= 9 && log2_size <= 16, "FSE Table must be between 2^9 to 2^16");
        // We need to know
        let mut cumul = [0u32; 256];

        todo!();

        // First, we build up the start positions of each symbol in the "cumul"
        // array. We special-case handle low-probability values by pre-filling
        // the symbol table from the highest spot down with that symbol's index.
        // This means weighting the low-prob symbols towards the end of the
        // state table.
        // /* symbol start positions */
        // // Figure out the start positions by just adding the counters up. We special
        // // case handle the low-probability symbols during table construction, by
        // {   U32 u;
        //     cumul[0] = 0;
        //     for (u=1; u <= maxSymbolValue+1; u++) {
        //         if (normalizedCounter[u-1]==-1) {  /* Low proba symbol */
        //             cumul[u] = cumul[u-1] + 1;
        //             tableSymbol[highThreshold--] = (FSE_FUNCTION_TYPE)(u-1);
        //         } else {
        //             cumul[u] = cumul[u-1] + normalizedCounter[u-1];
        //     }   }
        //     cumul[maxSymbolValue+1] = tableSize+1;
        // }

        // Next, we spread the symbols across the symbol table.
        //
        // Our step is "5/8 * size + 3", which this does with a pair of shifts and adds. Ok.
        // We start at position 0. Then, for each count in the normalized
        // histogram, we fill in the symbol value at our current position, then
        // increment the position pointer by our step. When the position pointer
        // is within the table but inside the "low probability area", we
        // increment a second time. The low-prob area is never more than 256 in
        // size (at max it's maybe 255?), so this is fine so long as the symbol table is at least 512 in size.
        // Via...magic... this uniquely steps through all locations in the
        // entire table. Fucked if I know how though.
        // /* Spread symbols */
        // {   U32 position = 0;
        //     U32 symbol;
        //     for (symbol=0; symbol<=maxSymbolValue; symbol++) {
        //         int nbOccurrences;
        //         int const freq = normalizedCounter[symbol];
        //         for (nbOccurrences=0; nbOccurrences<freq; nbOccurrences++) {
        //             tableSymbol[position] = (FSE_FUNCTION_TYPE)symbol;
        //             position = (position + step) & tableMask;
        //             while (position > highThreshold)
        //                 position = (position + step) & tableMask;   /* Low proba area */
        //     }   }

        //     assert(position==0);  /* Must have initialized all positions */
        // }

        // After we spread the symbols, it's time to build the actual encoding tables.
        // For each point in the table, we look up what the symbol value at that
        // spot is. We then write in the next state value, which is...
        // tableSize+u? and increment the table offset for that particular symbol.
        // /* Build table */
        // {   U32 u; for (u=0; u<tableSize; u++) {
        //     FSE_FUNCTION_TYPE s = tableSymbol[u];   /* note : static analyzer may not understand tableSymbol is properly initialized */
        //     tableU16[cumul[s]++] = (U16) (tableSize+u);   /* TableU16 : sorted by symbol order; gives next state value */
        // }   }

        // Final stretch. Here we build out the transform table. Not really sure what's happening here yet.
        // It looks like we fill in the number of bits & the next state
        // /* Build Symbol Transformation Table */
        //{   unsigned total = 0;
        //    unsigned s;
        //    for (s=0; s<=maxSymbolValue; s++) {
        //        switch (normalizedCounter[s])
        //        {
        //        case  0:
        //            /* filling nonetheless, for compatibility with FSE_getMaxNbBits() */
        //            symbolTT[s].deltaNbBits = ((tableLog+1) << 16) - (1<<tableLog);
        //            break;

        //        case -1:
        //        case  1:
        //            symbolTT[s].deltaNbBits = (tableLog << 16) - (1<<tableLog);
        //            symbolTT[s].deltaFindState = total - 1;
        //            total ++;
        //            break;
        //        default :
        //            {
        //                U32 const maxBitsOut = tableLog - BIT_highbit32 (normalizedCounter[s]-1);
        //                U32 const minStatePlus = normalizedCounter[s] << maxBitsOut;
        //                symbolTT[s].deltaNbBits = (maxBitsOut << 16) - minStatePlus;
        //                symbolTT[s].deltaFindState = total - normalizedCounter[s];
        //                total +=  normalizedCounter[s];
        //}   }   }   }

/*
size_t FSE_buildCTable_wksp(FSE_CTable* ct,
                      const short* normalizedCounter, unsigned maxSymbolValue, unsigned tableLog,
                            void* workSpace, size_t wkspSize)
{
    U32 const tableSize = 1 << tableLog;
    U32 const tableMask = tableSize - 1;
    void* const ptr = ct;
    U16* const tableU16 = ( (U16*) ptr) + 2;
    void* const FSCT = ((U32*)ptr) + 1 /* header */ + (tableLog ? tableSize>>1 : 1) ;
    FSE_symbolCompressionTransform* const symbolTT = (FSE_symbolCompressionTransform*) (FSCT);
    U32 const step = FSE_TABLESTEP(tableSize);
    U32 cumul[FSE_MAX_SYMBOL_VALUE+2];

    FSE_FUNCTION_TYPE* const tableSymbol = (FSE_FUNCTION_TYPE*)workSpace;
    U32 highThreshold = tableSize-1;

    /* CTable header */
    if (((size_t)1 << tableLog) * sizeof(FSE_FUNCTION_TYPE) > wkspSize) return ERROR(tableLog_tooLarge);
    tableU16[-2] = (U16) tableLog;
    tableU16[-1] = (U16) maxSymbolValue;
    assert(tableLog < 16);   /* required for threshold strategy to work */

    /* For explanations on how to distribute symbol values over the table :
     * http://fastcompression.blogspot.fr/2014/02/fse-distributing-symbol-values.html */

     #ifdef __clang_analyzer__
     memset(tableSymbol, 0, sizeof(*tableSymbol) * tableSize);   /* useless initialization, just to keep scan-build happy */
     #endif

    /* symbol start positions */
    // Figure out the start positions by just adding the counters up. We special
    // case handle the low-probability symbols during table construction, by
    {   U32 u;
        cumul[0] = 0;
        for (u=1; u <= maxSymbolValue+1; u++) {
            if (normalizedCounter[u-1]==-1) {  /* Low proba symbol */
                cumul[u] = cumul[u-1] + 1;
                tableSymbol[highThreshold--] = (FSE_FUNCTION_TYPE)(u-1);
            } else {
                cumul[u] = cumul[u-1] + normalizedCounter[u-1];
        }   }
        cumul[maxSymbolValue+1] = tableSize+1;
    }

    /* Spread symbols */
    {   U32 position = 0;
        U32 symbol;
        for (symbol=0; symbol<=maxSymbolValue; symbol++) {
            int nbOccurrences;
            int const freq = normalizedCounter[symbol];
            for (nbOccurrences=0; nbOccurrences<freq; nbOccurrences++) {
                tableSymbol[position] = (FSE_FUNCTION_TYPE)symbol;
                position = (position + step) & tableMask;
                while (position > highThreshold)
                    position = (position + step) & tableMask;   /* Low proba area */
        }   }

        assert(position==0);  /* Must have initialized all positions */
    }

    /* Build table */
    {   U32 u; for (u=0; u<tableSize; u++) {
        FSE_FUNCTION_TYPE s = tableSymbol[u];   /* note : static analyzer may not understand tableSymbol is properly initialized */
        tableU16[cumul[s]++] = (U16) (tableSize+u);   /* TableU16 : sorted by symbol order; gives next state value */
    }   }

    /* Build Symbol Transformation Table */
    {   unsigned total = 0;
        unsigned s;
        for (s=0; s<=maxSymbolValue; s++) {
            switch (normalizedCounter[s])
            {
            case  0:
                /* filling nonetheless, for compatibility with FSE_getMaxNbBits() */
                symbolTT[s].deltaNbBits = ((tableLog+1) << 16) - (1<<tableLog);
                break;

            case -1:
            case  1:
                symbolTT[s].deltaNbBits = (tableLog << 16) - (1<<tableLog);
                symbolTT[s].deltaFindState = total - 1;
                total ++;
                break;
            default :
                {
                    U32 const maxBitsOut = tableLog - BIT_highbit32 (normalizedCounter[s]-1);
                    U32 const minStatePlus = normalizedCounter[s] << maxBitsOut;
                    symbolTT[s].deltaNbBits = (maxBitsOut << 16) - minStatePlus;
                    symbolTT[s].deltaFindState = total - normalizedCounter[s];
                    total +=  normalizedCounter[s];
    }   }   }   }

    return 0;
}
*/
    }
}