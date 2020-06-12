import h5py

def smooth_over(list_to_smooth, smooth_last):
    smoothed = [list_to_smooth[0]]
    for i in range(1, len(list_to_smooth)+1):
        if i < smooth_last:
            smoothed.append(
                sum(list_to_smooth[:i]) / len(list_to_smooth[:i]))
        else:
            assert smooth_last == len(list_to_smooth[i-smooth_last:i])
            smoothed.append(
                sum(list_to_smooth[i-smooth_last:i]) / smooth_last
                )
    return smoothed

def add_to_h5(location, data_dict, group_name='custom_group'):

    with h5py.File(location, 'a') as hf:

        group = hf.create_group(group_name)

        for k, v in data_dict.items():
            group.create_dataset(k, data=v)
