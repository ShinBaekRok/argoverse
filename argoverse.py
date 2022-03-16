#!/usr/bin/env python
# coding: utf-8

# In[1544]:


import argoverse
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from argoverse.map_representation.map_api import ArgoverseMap
import seaborn as sns
import random
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as pyo
pyo.init_notebook_mode()


# In[2]:


#Define viz_sequence_inter
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp

from argoverse.map_representation.map_api import ArgoverseMap

_ZORDER = {"AGENT": 15, "AV": 10, "OTHERS": 5}


def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if polyline.shape[0] - len(duplicates) < 4:
        return polyline
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))


def viz_sequence_inter(
    df: pd.DataFrame,
    lane_centerlines: Optional[List[np.ndarray]] = None,
    show: bool = True,
    smoothen: bool = False,
) -> None:

    # Seq data
    city_name = df["CITY_NAME"].values[0]

    if lane_centerlines is None:
        # Get API for Argo Dataset map
        avm = ArgoverseMap()
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    plt.figure(0, figsize=(8, 7))

    x_min = min(df["X"])
    x_max = max(df["X"])
    y_min = min(df["Y"])
    y_max = max(df["Y"])

    if lane_centerlines is None:

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        lane_centerlines = []
        # Get lane centerlines which lie within the range of trajectories
        for lane_id, lane_props in seq_lane_props.items():

            lane_cl = lane_props.centerline

            if (
                np.min(lane_cl[:, 0]) < x_max
                and np.min(lane_cl[:, 1]) < y_max
                and np.max(lane_cl[:, 0]) > x_min
                and np.max(lane_cl[:, 1]) > y_min
            ):
                lane_centerlines.append(lane_cl)

    for lane_cl in lane_centerlines:
        plt.plot(
            lane_cl[:, 0],
            lane_cl[:, 1],
            "--",
            color="grey",
            alpha=1,
            linewidth=1,
            zorder=0,
        )
    frames = df.groupby("TRACK_ID")

    plt.xlabel("Map X")
    plt.ylabel("Map Y")

    color_dict = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}
    object_type_tracker: Dict[int, int] = defaultdict(int)

    # Plot all the tracks up till current frame
    for group_name, group_data in frames:
        object_type = group_data["OBJECT_TYPE"].values[0]

        cor_x = group_data["X"].values
        cor_y = group_data["Y"].values

        if smoothen:
            polyline = np.column_stack((cor_x, cor_y))
            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)
            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]

        plt.plot(
            cor_x,
            cor_y,
            "-",
            color=color_dict[object_type],
            label=object_type if not object_type_tracker[object_type] else "",
            alpha=1,
            linewidth=1,
            zorder=_ZORDER[object_type],
        )
        

        
        final_x = cor_x[-1]
        final_y = cor_y[-1]
    
        if object_type == "AGENT":
            marker_type = "o"
            marker_size = 7
        elif object_type == "OTHERS":
            marker_type = "o"
            marker_size = 7
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 7
        if object_type =='AGENT':
            x_inter = cor_x[20]
            y_inter = cor_y[20]
            
            plt.plot(
                x_inter,
                y_inter,
                marker_type,
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=.5,
                markersize=marker_size,
                zorder=_ZORDER[object_type],
            )
        
        
        plt.plot(
            final_x,
            final_y,
            marker_type,
            color=color_dict[object_type],
            label=object_type if not object_type_tracker[object_type] else "",
            alpha=1,
            markersize=marker_size,
            zorder=_ZORDER[object_type],
        )

        object_type_tracker[object_type] += 1

    red_star = mlines.Line2D([], [], color="red", marker="*", linestyle="None", markersize=7, label="Agent")
    green_circle = mlines.Line2D(
        [],
        [],
        color="green",
        marker="o",
        linestyle="None",
        markersize=7,
        label="Others",
    )
    black_triangle = mlines.Line2D([], [], color="black", marker="^", linestyle="None", markersize=7, label="AV")

    plt.axis("off")
    if show:
        plt.show()


# In[3]:


root_dir = '/Users/shin1/argoverse-api/train/data'


# In[4]:


afl = ArgoverseForecastingLoader(root_dir)
am = ArgoverseMap()


# In[5]:


print('Total number of sequences:', len(afl))


# In[6]:


MIA_agent = []
MIA_raw = []
MIA_num_tracks = []
PIT_agent = []
PIT_raw = []
PIT_num_tracks = []

for data in afl:
    if data.city == 'MIA':
        MIA_agent.append(data.agent_traj)
        MIA_raw.append(data.seq_df)
        MIA_num_tracks.append(data.num_tracks)
    else:
        PIT_agent.append(data.agent_traj)
        PIT_raw.append(data.seq_df)
        PIT_num_tracks.append(data.num_tracks)


# In[1176]:


print('Total number of sequences:', len(afl))
print('\nNumber of MIA data:',len(MIA_raw))
print('Number of PIT data:',len(PIT_raw))


# In[1541]:


def get_velocity(raw_data):
    if raw_data == MIA_raw:
        agent = MIA_agent
    else:
        agent = PIT_agent
        
    diff = []
    vel = []
    for idx, xy_points in enumerate(agent):
        xy_df = pd.DataFrame(xy_points)
        
        xy_difference = (xy_df - xy_df.shift(1))[1:]
        
        time_df = pd.DataFrame(np.unique(raw_data[idx]['TIMESTAMP']))
        
        time_difference = (time_df - time_df.shift(1))[1:]
        
        velocity = np.array(1/t)[:,0] * np.sqrt(np.sum((xy_difference)**2, axis=1))
                                                
        mean_velocity = np.mean(velocity)
        
        diff.append(velocity)
        vel.append(mean_velocity)

    return diff, vel


# In[1616]:


MIA_diff, MIA_velocity = get_velocity(MIA_raw)
PIT_diff, PIT_velocity = get_velocity(PIT_raw)
print('MIA:\nmean velocity(mpt): %.3f, std: %.3f, mean number of tracks: %d\n' 
      %(np.mean(MIA_velocity), np.std(MIA_velocity), np.mean(MIA_num_tracks)))
print('PIT:\nmean velocity(mpt): %.3f, std: %.3f, mean number of tracks: %d' 
      %(np.mean(PIT_velocity), np.std(PIT_velocity), np.mean(PIT_num_tracks)))


# In[1590]:


pio.templates.default = 'simple_white'
mean_velocity_df = pd.concat([pd.DataFrame({'Mean Velocity':MIA_velocity, 'City':'MIA'}),
                              pd.DataFrame({'Mean Velocity':PIT_velocity, 'City':'PIT'})])
px.violin(mean_velocity_df, x='City', y='Mean Velocity', box = True, color='City')


# In[1617]:


plt.figure(figsize=(8,6))
sns.kdeplot(MIA_velocity,color='blue', shade=True, label='MIA')
sns.kdeplot(PIT_velocity,color='orangered', shade=True, label='PIT')
plt.legend(title="Velocity")
plt.xlabel('Meter/Timestep')
plt.show()


# In[1619]:


import scipy.stats

scipy.stats.ttest_ind(MIA_velocity, PIT_velocity, equal_var=False)


# In[1452]:


#Outlier
i = -1
for idx in MIA_velocity:
    i+=1
    if idx>80:
        print(i)


# In[1484]:


viz_sequence(MIA_raw[1])


# In[1453]:


viz_sequence(MIA_raw[5445],show=True)
print(MIA_velocity[5445])


# In[1454]:


viz_sequence(MIA_raw[36931],show=True)
print(MIA_velocity[36931])


# In[15]:


#Outlier
i = -1
for idx in PIT_velocity:
    i+=1
    if idx>120:
        print(i)


# In[16]:


viz_sequence(PIT_raw[42194],show=True)
print(PIT_velocity[42194])


# In[1624]:


num_tracks_df = pd.concat([pd.DataFrame({'number of tracks':MIA_num_tracks, 'City':'MIA'}),
                           pd.DataFrame({'number of tracks':PIT_num_tracks, 'City':'PIT'})])
px.violin(num_tracks_df, x='City', y='number of tracks', box = True, color='City')


# In[1534]:


print(np.corrcoef(MIA_num_tracks, MIA_velocity)[0,1])
print(np.corrcoef(PIT_num_tracks, PIT_velocity)[0,1])


# In[1626]:


MIA_dic = {'City':'MIA', 'num_tracks':MIA_num_tracks, 'velocity':MIA_velocity}
PIT_dic = {'City':'PIT', 'num_tracks':PIT_num_tracks, 'velocity':PIT_velocity}


# In[1627]:


track_vel = pd.concat([pd.DataFrame(MIA_dic),pd.DataFrame(PIT_dic)])


# In[1628]:


fig = px.scatter(track_vel, x='num_tracks', y='velocity', color='City' ,template='simple_white')
fig.update_traces(marker=dict(opacity=0.5))


# In[691]:


def intersection(agent_xy, city_name):
    is_in_intersection=[]
    
    for time in range(50):
        lane_segments = am.get_lane_segments_containing_xy(agent_xy[time][0],agent_xy[time][1], city_name=city_name)
        in_intersection=[]

        for idx,lane_id in enumerate(lane_segments):
            in_intersection.append(am.lane_is_in_intersection(lane_segment_id=lane_id, city_name=city_name))
        
        if len(in_intersection)==0:
            is_in_intersection.append(False)
        elif len(np.where(np.array(in_intersection)==True)[0])/len(in_intersection)<0.5:
            is_in_intersection.append(False)
        else:
            is_in_intersection.append(True)

    return is_in_intersection


# In[692]:


MIA_intersection = []
for agent in tqdm.tqdm(MIA_agent): 
    MIA_intersection.append(intersection(agent, city_name='MIA'))


# In[693]:


PIT_intersection = []
for agent in tqdm.tqdm(PIT_agent):
    PIT_intersection.append(intersection(agent, city_name='PIT'))


# In[710]:


def intersection_idx(intersection):
    intersection_idx = []
    no_intersection_idx = []
    for idx, intersect in enumerate(intersection):
        if np.any(np.array(intersect)==True):
            intersection_idx.append(idx)
        else:
            no_intersection_idx.append(idx)
    return intersection_idx, no_intersection_idx


# In[712]:


MIA_inter_idx, MIA_nointer_idx = intersection_idx(MIA_intersection)
PIT_inter_idx, PIT_nointer_idx = intersection_idx(PIT_intersection)


# In[713]:


print('MIA trajectory with intersection(%): {0:.2f}%'.format(len(MIA_inter_idx)/len(MIA_raw)*100))
print('PIT trajectory with intersection(%): {0:.2f}%'.format(len(PIT_inter_idx)/len(PIT_raw)*100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[901]:


def get_lane_direction(agent_xy, city_name):
    lane_segment = []
    lane_direction = []
    for i in range(0,50):
        turnlane = []
        lane_segment = am.get_lane_segments_containing_xy(agent_xy[i][0],agent_xy[i][1], city_name=city_name)

        for lane_id in lane_segment:
            turnlane.append(am.get_lane_turn_direction(lane_segment_id=lane_id,city_name=city_name))
        lane_direction.append(turnlane)
    return lane_direction


# In[750]:


MIA_intersection_5 = [] #0.5초 단위로 어떤 차선을 타고 있는지
for intersect in MIA_intersection:
    mean = []
    for i in range(0,46,5):
        if len(np.where(np.array(intersect[i:i+5])==True)[0])>3:
            mean.append(True)
        else:
            mean.append(False)
    MIA_intersection_5.append(mean)
    
PIT_intersection_5 = [] 
for intersect in PIT_intersection:
    mean = []
    for i in range(0,46,5):
        if len(np.where(np.array(intersect[i:i+5])==True)[0])>3:
            mean.append(True)
        else:
            mean.append(False)
    PIT_intersection_5.append(mean)


# In[1068]:


def left_lane(agent, city_name):
    leftlane = []
    
    for idx in tqdm.tqdm(range(len(agent))):
        lane_dir = get_lane_direction(agent[idx],city_name=city_name)
        for time in range(0,41):
            leftsig = list(np.repeat(False,10))
            for idxx, direction in enumerate(lane_dir[time:time+10]):
                if 'LEFT' in direction:
                    leftsig[idxx]=True
                
            if leftsig == list(np.repeat(True,10)):
                leftlane.append(idx)
                break

    return leftlane


# In[1071]:


def right_lane(agent, city_name):
    rightlane = []

    for idx in tqdm.tqdm(range(len(agent))):
        lane_dir = get_lane_direction(agent[idx],city_name=city_name)
        for time in range(0,46):
            rightsig = list(np.repeat(False,5))
            for idxx, direction in enumerate(lane_dir[time:time+5]):
                if 'RIGHT' in direction:
                    rightsig[idxx]=True

            if rightsig == list(np.repeat(True,5)):
                rightlane.append(idx)
                break
   
    return rightlane


# In[1069]:


MIA_leftlane = left_lane(MIA_agent, 'MIA')


# In[1072]:


MIA_rightlane = right_lane(MIA_agent, 'MIA')


# In[1074]:


print(len(MIA_leftlane), len(MIA_rightlane))


# In[1070]:


PIT_leftlane = left_lane(PIT_agent, 'PIT')


# In[1073]:


PIT_rightlane = right_lane(PIT_agent, 'PIT')


# In[1075]:


print(len(PIT_leftlane), len(PIT_rightlane))


# In[1080]:


def get_turn(raw_data, threshold=0.4):
    left_turn = []
    right_turn = []
    forward_inter = []
    
    city_name = raw_data[0]['CITY_NAME'][0]
    
    if city_name == 'MIA':
        xycoord = MIA_agent
        nointer_idx = MIA_nointer_idx
        leftlane = MIA_leftlane
        rightlane = MIA_rightlane
    else:
        xycoord = PIT_agent
        nointer_idx = PIT_nointer_idx
        leftlane = PIT_leftlane
        rightlane = PIT_rightlane

    for idx, data in enumerate(xycoord):

        if idx in nointer_idx:
            continue
            
        temp_x1 = []
        temp_y1 = []
        temp_x2 = []
        temp_y2 = []

        for xy in data[:3]:
            temp_x1.append(xy[0])
            temp_y1.append(xy[1])
        x1 = np.mean(temp_x1)
        y1 = np.mean(temp_y1)
        
        for xy in data[3:6]:
            temp_x2.append(xy[0])
            temp_y2.append(xy[1])
        x2 = np.mean(temp_x2)
        y2 = np.mean(temp_y2)

        v1 = (x2-x1, y2-y1)
        
        temp_x1 = []
        temp_y1 = []
        temp_x2 = []
        temp_y2 = []

        for xy in data[44:47]:
            temp_x1.append(xy[0])
            temp_y1.append(xy[1])
        x1 = np.mean(temp_x1)
        y1 = np.mean(temp_y1)
        
        for xy in data[47:50]:
            temp_x2.append(xy[0])
            temp_y2.append(xy[1])
        x2 = np.mean(temp_x2)
        y2 = np.mean(temp_y2)

        v2 = (x2-x1, y2-y1)

        if (v1==(0,0))|(v2==(0,0)):
            pass
        
        else:
            theta = np.arcsin((v1[0]*v2[1]-v1[1]*v2[0])/(np.sqrt(v1[0]**2+v1[1]**2)*np.sqrt(v2[0]**2+v2[1]**2)))

            if (theta>threshold) & (idx in leftlane):
                left_turn.append(idx)
                
            elif (theta<-threshold) & (idx in rightlane):
                right_turn.append(idx)
            else:       
                forward_inter.append(idx)

    return left_turn, right_turn, nointer_idx, forward_inter


# In[1079]:


MIA_left, MIA_right, MIA_forward_nointer, MIA_forward_inter = get_turn(MIA_raw)
print('Percentage of MIA forward with no intersection:',round(len(MIA_forward_nointer)/len(MIA_agent)*100,3))
print('Percentage of MIA forward with intersection:',round(len(MIA_forward_inter)/len(MIA_agent)*100,3))
print('Percentage of MIA left:',round(len(MIA_left)/len(MIA_agent)*100,3))
print('Percentage of MIA right:',round(len(MIA_right)/len(MIA_agent)*100,3))


# In[1081]:


random.seed(0)
random.sample(MIA_left,3)


# In[1082]:


viz_sequence(MIA_raw[95797])


# In[1083]:


viz_sequence(MIA_raw[43820])


# In[1084]:


viz_sequence(MIA_raw[86185])


# In[1346]:


random.seed(0)
random.sample(MIA_forward, 3)


# In[1347]:


viz_sequence(MIA_raw[66659])


# In[1348]:


viz_sequence(MIA_raw[72813])


# In[1349]:


viz_sequence(MIA_raw[6961])


# In[1085]:


random.seed(0)
random.sample(MIA_forward_nointer, 3)


# In[720]:


viz_sequence(MIA_raw[52780])


# In[721]:


viz_sequence(MIA_raw[105058])


# In[722]:


viz_sequence(MIA_raw[57654])


# In[1088]:


PIT_left, PIT_right, PIT_forward_nointer, PIT_forward_inter = get_turn(PIT_raw)
print('Percentage of PIT forward with no intersection:',round(len(PIT_forward_nointer)/len(PIT_agent)*100,3))
print('Percentage of PIT forward with intersection:',round(len(PIT_forward_inter)/len(PIT_agent)*100,3))
print('Percentage of PIT left:',round(len(PIT_left)/len(PIT_agent)*100,3))
print('Percentage of PIT right:',round(len(PIT_right)/len(PIT_agent)*100,3))


# In[1089]:


random.seed(0)
random.sample(PIT_right, 3)


# In[1090]:


viz_sequence(PIT_raw[50974])


# In[1091]:


viz_sequence(PIT_raw[55953])


# In[1092]:


viz_sequence(PIT_raw[5511])


# In[1093]:


random.seed(0)
random.sample(PIT_forward_inter, 3)


# In[1094]:


viz_sequence(PIT_raw[82048])


# In[1095]:


viz_sequence(PIT_raw[37402])


# In[1096]:


viz_sequence(PIT_raw[74635])


# In[1097]:


MIA_forward = np.union1d(MIA_forward_nointer, MIA_forward_inter)
PIT_forward = np.union1d(PIT_forward_nointer, PIT_forward_inter)


# In[1697]:


fig = plotly.subplots.make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                                   subplot_titles=['MIA', 'PIT'])
fig.add_trace(go.Pie(labels=['left','forward(inter)','forward(no inter)','right'], values=[len(MIA_left),len(MIA_forward_inter),len(MIA_forward_nointer), len(MIA_right)],textinfo='label+percent', name='Trajectory Type')
              ,1,1)
fig.add_trace(go.Pie(labels=['left','forward(inter)','forward(no inter)','right'], values=[len(PIT_left),len(PIT_forward_inter),len(PIT_forward_nointer),len(PIT_right)], textinfo='label+percent',name='Trajectory Type')
              ,1,2)
fig.update_layout(title_text='Percentage of trajectory type by city')
fig.show()


# In[1099]:


def get_elements_in_list(in_list, in_indices):
    return [in_list[i] for i in in_indices]


# In[1100]:


def get_velocity_idx(diff, index):
    vel = []
    for element in get_elements_in_list(diff, index):
        vel.append(np.mean(element))
    return vel


# In[1658]:


MIA_forward_vel = np.mean(get_velocity_idx(MIA_diff, MIA_forward)), np.std(get_velocity_idx(MIA_diff, MIA_forward))
MIA_forward_nointer_vel = np.mean(get_velocity_idx(MIA_diff,MIA_forward_nointer)), np.std(get_velocity_idx(MIA_diff,MIA_forward_nointer))
MIA_forward_inter_vel = np.mean(get_velocity_idx(MIA_diff,MIA_forward_inter)), np.std(get_velocity_idx(MIA_diff,MIA_forward_inter))
MIA_left_vel = np.mean(get_velocity_idx(MIA_diff,MIA_left)), np.std(get_velocity_idx(MIA_diff,MIA_left))
MIA_right_vel = np.mean(get_velocity_idx(MIA_diff,MIA_right)), np.std(get_velocity_idx(MIA_diff,MIA_right))

print('Velocity of MIA forward:\n mean:%.3f std:%.3f'%(MIA_forward_vel[0], MIA_forward_vel[1]))
print('Velocity of MIA forward with no intersection:\n mean:%.3f std:%.3f'%(MIA_forward_nointer_vel[0],MIA_forward_nointer_vel[1]))
print('Velocity of MIA forward with intersection:\n mean:%.3f std:%.3f'%(MIA_forward_inter_vel[0],MIA_forward_inter_vel[1]))
print('Velocity of MIA left:\n mean:%.3f std:%.3f'%(MIA_left_vel[0],MIA_left_vel[1]))
print('Velocity of MIA right:\n mean:%.3f std:%.3f'%(MIA_right_vel[0],MIA_right_vel[1]))


# In[1659]:


viz_sequence(MIA_raw[MIA_forward_nointer[2]],show=True)
#traffic jam 때문에 교차로를 들어가기 전부터 막힘 -->속도 낮음


# In[1660]:


PIT_forward_vel = np.mean(get_velocity_idx(PIT_diff, PIT_forward)), np.std(get_velocity_idx(PIT_diff, PIT_forward))
PIT_forward_nointer_vel = np.mean(get_velocity_idx(PIT_diff,PIT_forward_nointer)), np.std(get_velocity_idx(PIT_diff,PIT_forward_nointer))
PIT_forward_inter_vel = np.mean(get_velocity_idx(PIT_diff,PIT_forward_inter)), np.std(get_velocity_idx(PIT_diff,PIT_forward_inter))
PIT_left_vel = np.mean(get_velocity_idx(PIT_diff,PIT_left)), np.std(get_velocity_idx(PIT_diff,PIT_left))
PIT_right_vel = np.mean(get_velocity_idx(PIT_diff,PIT_right)), np.std(get_velocity_idx(PIT_diff,PIT_right))

print('Velocity of PIT forward:\n mean:%.3f std:%.3f'%(PIT_forward_vel[0], PIT_forward_vel[1]))
print('Velocity of PIT forward with no intersection:\n mean:%.3f std:%.3f'%(PIT_forward_nointer_vel[0],PIT_forward_nointer_vel[1]))
print('Velocity of PIT forward with intersection:\n mean:%.3f std:%.3f'%(PIT_forward_inter_vel[0],PIT_forward_inter_vel[1]))
print('Velocity of PIT left:\n mean:%.3f std:%.3f'%(PIT_left_vel[0],PIT_left_vel[1]))
print('Velocity of PIT right:\n mean:%.3f std:%.3f'%(PIT_right_vel[0],PIT_right_vel[1]))


# In[1672]:


scipy.stats.ttest_ind(get_velocity_idx(PIT_diff, PIT_right), get_velocity_idx(MIA_diff, MIA_right),equal_var=False)


# In[1661]:


MIA_forward_vel = pd.DataFrame({'Mean Velocity':get_velocity_idx(MIA_diff, MIA_forward), 
                                'City':'MIA', 'Trajectory_type':'forward'})
MIA_left_vel = pd.DataFrame({'Mean Velocity':get_velocity_idx(MIA_diff,MIA_left), 
                             'City':'MIA', 'Trajectory_type':'left'})
MIA_right_vel = pd.DataFrame({'Mean Velocity':get_velocity_idx(MIA_diff,MIA_right), 
                              'City':'MIA', 'Trajectory_type':'right'})

PIT_forward_vel = pd.DataFrame({'Mean Velocity':get_velocity_idx(PIT_diff, PIT_forward), 
                                'City':'PIT', 'Trajectory_type':'forward'})
PIT_left_vel = pd.DataFrame({'Mean Velocity':get_velocity_idx(PIT_diff,PIT_left), 
                             'City':'PIT', 'Trajectory_type':'left'})
PIT_right_vel = pd.DataFrame({'Mean Velocity':get_velocity_idx(PIT_diff,PIT_right), 
                              'City':'PIT', 'Trajectory_type':'right'})

velocity_df = pd.concat([MIA_forward_vel,  MIA_left_vel, MIA_right_vel,
                         PIT_forward_vel,  PIT_left_vel, PIT_right_vel])


# In[1662]:


pio.templates.default = 'simple_white'
px.box(data_frame = velocity_df
       ,x = 'Mean Velocity'
       ,y = 'Trajectory_type'
       ,color = 'City'
       )


# In[1693]:


MIA_forward_inter_vel = pd.DataFrame({'Mean Velocity':get_velocity_idx(MIA_diff, MIA_forward_inter),
                                      'City':'MIA', 'Trajectory_type':'forward(intersect)'})
MIA_forward_nointer_vel = pd.DataFrame({'Mean Velocity':get_velocity_idx(MIA_diff,MIA_forward_nointer), 
                                      'City':'MIA', 'Trajectory_type':'forward(no intersect)'})

PIT_forward_nointer_vel = pd.DataFrame({'Mean Velocity':get_velocity_idx(PIT_diff,PIT_forward_nointer), 
                                        'City':'PIT', 'Trajectory_type':'forward(no intersect)'})
PIT_forward_inter_vel = pd.DataFrame({'Mean Velocity':get_velocity_idx(PIT_diff,PIT_forward_inter), 
                                      'City':'PIT', 'Trajectory_type':'forward(intersect)'})


velocity_df = pd.concat([MIA_forward_nointer_vel, MIA_forward_inter_vel,
                         PIT_forward_nointer_vel, PIT_forward_inter_vel])


# In[1694]:


pio.templates.default = 'simple_white'
px.box(data_frame = velocity_df
       ,x = 'Mean Velocity'
       ,y = 'Trajectory_type'
       ,color = 'City'
       )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1699]:


MIA_intersection_velocity = MIA_forward_inter_vel['Mean Velocity']
PIT_intersection_velocity = PIT_forward_inter_vel['Mean Velocity']

MIA_no_intersection_velocity = MIA_forward_nointer_vel['Mean Velocity']
PIT_no_intersection_velocity = PIT_forward_nointer_vel['Mean Velocity']

MIA_left_velocity = MIA_left_vel['Mean Velocity']
PIT_left_velocity = PIT_left_vel['Mean Velocity']

MIA_right_velocity = MIA_right_vel['Mean Velocity']
PIT_right_velocity = PIT_right_vel['Mean Velocity']


# In[1704]:


plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
sns.kdeplot(MIA_intersection_velocity, color='blue', fill=True, label='MIA(Intersection)')
sns.kdeplot(PIT_intersection_velocity, color='orangered', fill=True, label='PIT(Intersection)')
plt.legend(title='Velocity of going straight')
plt.xlabel('meter/timestep')

plt.subplot(2,1,2)
sns.kdeplot(MIA_no_intersection_velocity,color='blue', fill=True, label='MIA(No Intersection)')
sns.kdeplot(PIT_no_intersection_velocity,color='orangered', fill=True, label='PIT(No Intersection)')
plt.legend(title="Velocity of going straight")
plt.xlabel('meter/timestep')

plt.show()
#NO intersection일 때 아마 PIT에서 훈련한 모델이 더 많이 나간다고 예측할 것


# In[1111]:


plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
sns.kdeplot(MIA_left_velocity, color='blue', fill=True, label='MIA')
sns.kdeplot(PIT_left_velocity, color='orangered', fill=True, label='PIT')
plt.legend(title='Velocity of turning left')
plt.xlabel('km/h')

plt.subplot(2,1,2)
sns.kdeplot(MIA_right_velocity,color='blue', fill=True, label='MIA')
sns.kdeplot(PIT_right_velocity,color='orangered', fill=True, label='PIT')
plt.legend(title="Velocity of turning right")
plt.xlabel('km/h')

plt.show()

#두 분포가 거의 비슷


# In[ ]:





# In[ ]:





# In[ ]:





# In[1112]:


def get_segment(agent_xy, city_name):
    lane_segment = []
    for i in range(0,50):

        temp_lane = am.get_lane_segments_containing_xy(agent_xy[i][0],agent_xy[i][1], city_name=city_name)
        lane_segment.append(temp_lane)

    return lane_segment


# In[1192]:


def changing_lane(forward_idx, city_name):

    changing_lane_id = []
    if city_name == 'MIA':
        agent = MIA_agent
    else:
        agent = PIT_agent
    
    for idx in tqdm.tqdm(forward_idx):
        forward_lane_id = []
        lane_seg = get_segment(agent[idx], city_name=city_name)
        for seg in lane_seg:
            for lane_id in seg:
                none_id = []
                lane_dir = am.get_lane_turn_direction(lane_segment_id=lane_id,city_name=city_name)
                
                if lane_dir == 'NONE':
                    none_id.append(lane_id)
                    
            forward_lane_id.append(none_id)
            
        for index, each_list in enumerate(forward_lane_id):
            for each_id in each_list:
                if index == 0:
                    adjacent_id = am.get_lane_segment_adjacent_ids(lane_segment_id = each_id, city_name=city_name)
                else:
                    if each_id in adjacent_id:
                        changing_lane_id.append(idx)
                    adjacent_id = am.get_lane_segment_adjacent_ids(lane_segment_id = each_id, city_name=city_name)
    
    return list(np.unique(changing_lane_id))


# In[1174]:


MIA_forward_changing_lane = changing_lane(MIA_forward, city_name = 'MIA')


# In[1198]:


len(MIA_forward_changing_lane)


# In[1209]:


PIT_forward_changing_lane = changing_lane(PIT_forward, city_name = 'PIT')


# In[1257]:


len(PIT_forward_changing_lane)


# In[1197]:


np.mean(get_velocity_idx(MIA_diff,MIA_forward_changing_lane))


# In[1258]:


np.mean(get_velocity_idx(PIT_diff,PIT_forward_changing_lane))


# In[1196]:


viz_sequence(MIA_raw[64])


# In[1263]:


len(np.intersect1d(PIT_forward_changing_lane,PIT_forward_inter))


# In[1265]:


len(np.intersect1d(PIT_forward_changing_lane,PIT_forward_nointer))


# In[1269]:


np.mean(get_velocity_idx(PIT_diff, np.intersect1d(PIT_forward_changing_lane,PIT_forward_nointer)))


# In[1270]:


np.mean(get_velocity_idx(MIA_diff, np.intersect1d(MIA_forward_changing_lane,MIA_forward_nointer)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1673]:


MIA_right_diff = pd.DataFrame(np.mean(get_elements_in_list(MIA_diff,MIA_right),axis=0),columns=['MIA'])
PIT_right_diff = pd.DataFrame(np.mean(get_elements_in_list(PIT_diff,PIT_right),axis=0),columns=['PIT'])
right_diff = pd.concat([MIA_right_diff, PIT_right_diff],axis=1)


# In[1674]:


right_diff.plot(figsize=(8,6))
plt.title('Velocity of rightturn by time')
plt.xlabel('time(0.1second)')
plt.ylabel('Velocity(km/h)')
plt.show()


# In[1675]:


MIA_left_diff = pd.DataFrame(np.mean(get_elements_in_list(MIA_diff,MIA_left),axis=0),columns=['MIA'])
PIT_left_diff = pd.DataFrame(np.mean(get_elements_in_list(PIT_diff,PIT_left),axis=0),columns=['PIT'])
left_diff = pd.concat([MIA_left_diff, PIT_left_diff],axis=1)


# In[1676]:


left_diff.plot(figsize=(8,6))
plt.title('Velocity of leftturn by time')
plt.xlabel('time(0.1second)')
plt.ylabel('Velocity(km/h)')
plt.show()


# In[1677]:


MIA_forward_inter_diff = pd.DataFrame(np.mean(get_elements_in_list(MIA_diff,MIA_forward_inter),axis=0),columns=['MIA'])
PIT_forward_inter_diff = pd.DataFrame(np.mean(get_elements_in_list(PIT_diff,PIT_forward_inter),axis=0),columns=['PIT'])
forward_inter_diff = pd.concat([MIA_forward_inter_diff, PIT_forward_inter_diff],axis=1)


# In[1678]:


forward_inter_diff.plot(figsize=(8,6))
plt.title('Velocity of going straight(with intersection)by time')
plt.xlabel('time(0.1second)')
plt.ylabel('Velocity(km/h)')
plt.show()


# In[1679]:


MIA_forward_nointer_diff = pd.DataFrame(np.mean(get_elements_in_list(MIA_diff,MIA_forward_nointer),axis=0),columns=['MIA'])
PIT_forward_nointer_diff = pd.DataFrame(np.mean(get_elements_in_list(PIT_diff,PIT_forward_nointer),axis=0),columns=['PIT'])
forward_nointer_diff = pd.concat([MIA_forward_nointer_diff, PIT_forward_nointer_diff],axis=1)


# In[1680]:


forward_nointer_diff.plot(figsize=(8,6))
plt.title('Velocity of going straight(no intersection)by time')
plt.xlabel('time(0.1second)')
plt.ylabel('Velocity(km/h)')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1705]:


def abrupt_acc_dec(differentiation, threshold_acc=10, threshold_dec=10):
    acceleration_idx = []
    deceleration_idx = []
    
    for idx, diff in enumerate(differentiation):
        a = np.mean(diff[:20])
        b = np.mean(diff[20:])
        if b-a>threshold_acc:
            acceleration_idx.append(idx)

        if a-b>threshold_dec:
            deceleration_idx.append(idx)
            
    return acceleration_idx, deceleration_idx


# In[1706]:


MIA_acc, MIA_dec = abrupt_acc_dec(MIA_diff)
PIT_acc, PIT_dec = abrupt_acc_dec(PIT_diff)


# In[1707]:


print('Number of Abrupt acc in MIA:', len(MIA_acc))
print('Number of Abrupt acc in PIT:', len(PIT_acc))


# In[1708]:


print('Number of Abrupt dec in MIA:', len(MIA_dec))
print('Number of Abrupt dec in PIT:', len(PIT_dec))


# In[1709]:


#MIA 운전자들이 훨씬 거칠게 운전 --> 급감, 급가속 PIT보다 유의미하게 많음


# In[1710]:


plt.figure(figsize=(8,6))
plt.bar([0,0.9], [len(MIA_acc),len(MIA_dec)], label='MIA', color='blue',width=0.2)
plt.bar([0.3,1.2], [len(PIT_acc),len(PIT_dec)], label='PIT', color='orangered',width=0.2)
plt.xticks([0.15,1.05],['Acceleration','Deceleration'])
plt.legend()
plt.ylabel('Number')
plt.title('Number of Abrupt Acc, Dec')
plt.show()


# In[1711]:


print(MIA_acc[:5])
print(MIA_dec[:5])


# In[1712]:


viz_sequence_inter(MIA_raw[228], show=True)


# In[1718]:


viz_sequence_inter(MIA_raw[2667], show=True)


# In[ ]:




