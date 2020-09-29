import pandas as pd
import glob
import os

cols = ['epoch_test_MSE', 'epoch_test_MAPE', 'epoch_test_U_THEIL', 'epoch_test_ARV', 'epoch_test_POCID_ret', 'epoch_test_SLG_ret', 'epoch_test_IA']

model_switcher = {'many-to-one_no-attention': 'MtO',
                  'many-to-many_no-attention': 'MtM',
                  'many-to-one_attention': 'MtO-A',
                  'many-to-many_attention': 'MtM-A',
                  'self_attn': 'self-attn'
                  }
for file_type in ['price', 'full']:
    df_all = pd.DataFrame()
    for coin in ['BTC', 'ETH', 'XMR', 'XRP', 'LTC', 'BCH', 'XLM', 'DASH', 'IOTA', 'TRX', 'DJIA']:
        for model in ['many-to-one_no-attention', 'many-to-many_no-attention', 'many-to-one_attention', 'many-to-many_attention', 'self_attn']:
            df_val = pd.DataFrame()
            for num_try in range(1, 6):
                    file = 'log'
                    files = sorted(glob.glob(r"insert log path here \logs\{}*{}*\*\*try_{}*{}.csv".format(model, coin, num_try, file)))
                    # Handle different files types
                    if file_type == 'price':
                        files = [file for file in files if not '_full_' in file]
                    elif file_type == 'full':
                        files = [file for file in files if '_full_' in file]

                    df = pd.read_csv(files[0], sep=";")
                    df_min_val = df.loc[df.val_mean_squared_error == df.val_mean_squared_error.min()]
                    min_epoch = df_min_val['epoch']
                    df_min_val = df_min_val.assign(**{'coin': coin, 'num_try': num_try, 'min_epoch': min_epoch})
                    df_min_val['min_epoch'] = min_epoch
                    file = 'metrics'
                    files = sorted(glob.glob(
                        r"insert log path here \many-to-one_attention*{}*\*\*try_{}*{}.csv".format(
                            coin, num_try, file)))
                    df_metrics = pd.read_csv(files[0], sep=";")
                    df_min_val = df_min_val.join(df_metrics, how= 'left')
                    df_val = pd.concat([df_val, df_min_val])

            df_means = pd.DataFrame(df_val[cols].mean(axis=0)).transpose()
            df_means = df_means.rename(columns={'epoch_test_MSE': 'MSE',
                                                'epoch_test_MAPE': 'MAPE',
                                                'epoch_test_U_THEIL': 'U THEIL',
                                                'epoch_test_ARV': 'ARV',
                                                'epoch_test_POCID_ret': 'POCID',
                                                'epoch_test_SLG_ret': 'SLG',
                                                'epoch_test_IA': 'IA'})
            df_means['Sample'] = coin
            df_means['Model'] = model_switcher.get(model)
            df_means['min_epoch'] = ', '.join(str(e) for e in df_val['min_epoch'].tolist())
            df_std = pd.DataFrame(df_val[cols].std(axis=0)).transpose()
            df_std = df_std.rename(columns={'epoch_test_MSE': 'MSE std',
                                            'epoch_test_MAPE': 'MAPE std',
                                            'epoch_test_U_THEIL': 'U THEIL std',
                                            'epoch_test_ARV': 'ARV std',
                                            'epoch_test_POCID_ret': 'POCID std',
                                            'epoch_test_SLG_ret': 'SLG std',
                                            'epoch_test_IA': 'IA std'})
            df_coin = df_means.join(df_std, how='left')
            df_coin['MSE raw'] = df_coin['MSE']
            df_coin['MSE'] = df_coin['MSE'] / 1e-5
            df_coin['MSE std'] = df_coin['MSE std'] / 1e-5
            df_coin['MSE * 10-5'] = df_coin['MSE'].map('{:.2f}'.format) + ' \pm ' + df_coin['MSE std'].map('{:.2f}'.format)
            df_coin['MAPE raw'] = df_coin['MAPE']
            df_coin['MAPE'] = df_coin['MAPE'] / 1e+4
            df_coin['MAPE std'] = df_coin['MAPE std'] / 1e+4
            df_coin['MAPE * 10+4'] = df_coin['MAPE'].map('{:.2f}'.format) + ' \pm ' + df_coin['MAPE std'].map('{:.2f}'.format)
            df_coin['U THEIL raw'] = df_coin['U THEIL']
            df_coin['U THEIL'] = df_coin['U THEIL'].map('{:.1f}'.format) + ' \pm ' + df_coin['U THEIL std'].map('{:.1f}'.format)
            df_coin['ARV raw'] = df_coin['ARV']
            df_coin['ARV'] = df_coin['ARV'].map('{:.1f}'.format) + ' \pm ' + df_coin['ARV std'].map('{:.1f}'.format)
            df_coin['POCID raw'] = df_coin['POCID']
            df_coin['POCID'] = df_coin['POCID'].map('{:.1f}'.format) + ' \pm ' + df_coin['POCID std'].map('{:.1f}'.format)
            df_coin['SLG raw'] = df_coin['SLG']
            df_coin['SLG'] = df_coin['SLG'] / 1e-5
            df_coin['SLG std'] = df_coin['SLG std'] / 1e-5
            df_coin['SLG * 10-5'] = df_coin['SLG'].map('{:.2f}'.format) + ' \pm ' + df_coin['SLG std'].map('{:.2f}'.format)
            df_coin['IA raw'] = df_coin['IA']
            df_coin['IA'] = df_coin['IA'].map('{:.2f}'.format) + ' \pm ' + df_coin['IA std'].map('{:.2f}'.format)

            df_all = pd.concat([df_all, df_coin])

    df_all_sum = df_all.groupby(['Model'], as_index=False)['MSE raw', 'MAPE raw', 'U THEIL raw', 'ARV raw', 'POCID raw', 'SLG raw', 'IA raw'].mean()
    df_all_sum['Sample'] = 'ALL'
    df_all_sum = df_all_sum.rename(columns={'MSE raw': 'MSE',
                                            'MAPE raw': 'MAPE',
                                            'U THEIL raw': 'U THEIL',
                                            'ARV raw': 'ARV',
                                            'POCID raw': 'POCID',
                                            'SLG raw': 'SLG',
                                            'IA raw': 'IA'})
    df_all_sum_std = df_all.groupby(['Model'])['MSE raw', 'MAPE raw', 'U THEIL raw', 'ARV raw', 'POCID raw', 'SLG raw', 'IA raw'].std()
    df_all_sum_std = df_all_sum_std.rename(columns={'MSE raw': 'MSE std',
                                                    'MAPE raw': 'MAPE std',
                                                    'U THEIL raw': 'U THEIL std',
                                                    'ARV raw': 'ARV std',
                                                    'POCID raw': 'POCID std',
                                                    'SLG raw': 'SLG std',
                                                    'IA raw': 'IA std'})
    df_all_sum_std.reset_index(inplace=True)
    df_all_sum = df_all_sum.merge(df_all_sum_std, on=['Model'], how='left')
    df_all_sum['MSE raw'] = df_all_sum['MSE']
    df_all_sum['MSE'] = df_all_sum['MSE'] / 1e-5
    df_all_sum['MSE std'] = df_all_sum['MSE std'] / 1e-5
    df_all_sum['MSE * 10-5'] = df_all_sum['MSE'].map('{:.2f}'.format) + ' \pm ' + df_all_sum['MSE std'].map('{:.2f}'.format)
    df_all_sum['MAPE raw'] = df_all_sum['MAPE']
    df_all_sum['MAPE'] = df_all_sum['MAPE'] / 1e+4
    df_all_sum['MAPE std'] = df_all_sum['MAPE std'] / 1e+4
    df_all_sum['MAPE * 10+4'] = df_all_sum['MAPE'].map('{:.2f}'.format) + ' \pm ' + df_all_sum['MAPE std'].map('{:.2f}'.format)
    df_all_sum['U THEIL raw'] = df_all_sum['U THEIL']
    df_all_sum['U THEIL'] = df_all_sum['U THEIL'].map('{:.1f}'.format) + ' \pm ' + df_all_sum['U THEIL std'].map('{:.1f}'.format)
    df_all_sum['ARV raw'] = df_all_sum['ARV']
    df_all_sum['ARV'] = df_all_sum['ARV'].map('{:.1f}'.format) + ' \pm ' + df_all_sum['ARV std'].map('{:.1f}'.format)
    df_all_sum['POCID raw'] = df_all_sum['POCID']
    df_all_sum['POCID'] = df_all_sum['POCID'].map('{:.1f}'.format) + ' \pm ' + df_all_sum['POCID std'].map('{:.1f}'.format)
    df_all_sum['SLG raw'] = df_all_sum['SLG']
    df_all_sum['SLG'] = df_all_sum['SLG'] / 1e-5
    df_all_sum['SLG std'] = df_all_sum['SLG std'] / 1e-5
    df_all_sum['SLG * 10-5'] = df_all_sum['SLG'].map('{:.2f}'.format) + ' \pm ' + df_all_sum['SLG std'].map('{:.2f}'.format)
    df_all_sum['IA raw'] = df_all_sum['IA']
    df_all_sum['IA'] = df_all_sum['IA'].map('{:.2f}'.format) + ' \pm ' + df_all_sum['IA std'].map('{:.2f}'.format)
    df_all_sum['Model'] = pd.Categorical(df_all_sum['Model'], ['MtO', 'MtM', 'MtO-A', 'MtM-A', 'self-attn'])
    df_all_sum = df_all_sum.sort_values('Model')

    df_all_all_avg = df_all_sum.mean()
    df_all_all_std = df_all_sum.std()

    data = {'ALL': ['ALL',
                    'ALL',
                    '{:.2f}'.format(df_all_all_avg['MSE']) + ' \pm ' + '{:.2f}'.format(df_all_all_std['MSE']),
                    '{:.2f}'.format(df_all_all_avg['MAPE']) + ' \pm ' + '{:.2f}'.format(df_all_all_std['MAPE']),
                    '{:.2f}'.format(df_all_all_avg['U THEIL raw']) + ' \pm ' + '{:.2f}'.format(
                        df_all_all_std['U THEIL raw']),
                    '{:.2f}'.format(df_all_all_avg['ARV raw']) + ' \pm ' + '{:.2f}'.format(df_all_all_std['ARV raw']),
                    '{:.2f}'.format(df_all_all_avg['POCID raw']) + ' \pm ' + '{:.2f}'.format(
                        df_all_all_std['POCID raw']),
                    '{:.2f}'.format(df_all_all_avg['SLG']) + ' \pm ' + '{:.2f}'.format(df_all_all_std['SLG']),
                    '{:.2f}'.format(df_all_all_avg['IA raw']) + ' \pm ' + '{:.2f}'.format(df_all_all_std['IA raw'])
                    ], }
    df_all_all = pd.DataFrame.from_dict(data, orient='index',
                                        columns=['Sample', 'Model', 'MSE * 10-5', 'MAPE * 10+4', 'U THEIL', 'ARV',
                                                 'POCID', 'SLG * 10-5', 'IA'])

    folder = r"insert target folder here"
    df_all = pd.concat([df_all, df_all_sum], sort=False)
    df_all = pd.concat([df_all, df_all_all], sort=False)
    file_name = "coin_evaluation_table_{}.xlsx".format(file_type)
    df_all.to_excel(os.path.join(folder, file_name))

    df_all = df_all[
        ['Sample', 'Model', 'MSE * 10-5', 'MAPE * 10+4', 'U THEIL', 'ARV', 'POCID', 'SLG * 10-5', 'IA']]
    file_name = "coin_evaluation_table_{}.tex".format(file_type)
    folder = r"insert target folder here"
    file_name = os.path.join(folder, file_name)
    df_all.to_latex(file_name, index=False, escape=True)
    import fileinput

    # Replace plus minus sign with latex symbol
    with fileinput.FileInput(file_name, inplace=True, backup='.bak') as file:
        i = 1
        for line in file:
            if i in ([*range(9, 59, 5)] +[64]):
                print(line.replace('\\textbackslash pm', '$\\pm$') + "\hline", end='')
            elif i == 59:
                print(line.replace('\\textbackslash pm', '$\\pm$')+ "\hline\hline", end='')
            else:
                print(line.replace('\\textbackslash pm', '$\\pm$'), end='')
            i += 1
