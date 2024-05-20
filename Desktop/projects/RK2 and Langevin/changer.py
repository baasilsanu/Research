#Changing df_3 into df_3_cleaned

df_3 = df_3[(df_3['epsilon'] != .152642) & (df_3['n_zero_square'] != 596.36)]

df_3 = df_3[(df_3['reversal_count'] > 0)]
df_3 = update_reversal_durations(df_3, 2000)
df_3['average_reversal_time'] = df_3['reversal_durations'].apply(calculate_average_reversal_time)
df_3['average_reversal_time_excluding_last'] = df_3['reversal_durations'].apply(calculate_average_reversal_time_excluding_last)

df_3_AnyLessThan5 = df_3['reversal_durations'].apply(any_values_less_than_5)
df_3['any_less_than_5'] = df_3_AnyLessThan5
df_3[(df_3['any_less_than_5'] == True)].head()
df_3_cleaned = df_3[(df_3['any_less_than_5'] == False)]
df_3_cleaned.head()