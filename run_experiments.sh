#for history in 1 5 9 13 17 21 25
for history in 9
do
   for future in 0 4 8 12 16 20 24
   do
      python3 data/prepare_data.py $history $future
      for channel in 0 1 2 3 4
      do
         python3 main.py LSTMClassifier $history $future $channel
      done
   done
done
