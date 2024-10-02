graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    high 100
    low 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "max_cpu"
    originator "cpu"
    owner "node"
    type "extrema"
  ]
  link_attrs_setting [
    distribution "uniform"
    dtype "int"
    generative 1
    high 100
    low 50
    name "bw"
    owner "link"
    type "resource"
  ]
  link_attrs_setting [
    name "max_bw"
    originator "bw"
    owner "link"
    type "extrema"
  ]
  link_attrs_setting [
    name "ltc"
    owner "link"
    type "latency"
    generative 1
    distribution "customized"
    max 1.0
    min 0.0
  ]
  save_dir "dataset/p_net"
  topology [
    type "waxman"
    wm_alpha 0.5
    wm_beta 0.2
  ]
  file_name "p_net.gml"
  num_nodes 100
  type "waxman"
  wm_alpha 0.5
  wm_beta 0.2
  node [
    id 0
    label "0"
    pos 0.8444218515250481
    pos 0.7579544029403025
    cpu 94
    max_cpu 94
  ]
  node [
    id 1
    label "1"
    pos 0.420571580830845
    pos 0.25891675029296335
    cpu 97
    max_cpu 97
  ]
  node [
    id 2
    label "2"
    pos 0.5112747213686085
    pos 0.4049341374504143
    cpu 50
    max_cpu 50
  ]
  node [
    id 3
    label "3"
    pos 0.7837985890347726
    pos 0.30331272607892745
    cpu 53
    max_cpu 53
  ]
  node [
    id 4
    label "4"
    pos 0.4765969541523558
    pos 0.5833820394550312
    cpu 53
    max_cpu 53
  ]
  node [
    id 5
    label "5"
    pos 0.9081128851953352
    pos 0.5046868558173903
    cpu 89
    max_cpu 89
  ]
  node [
    id 6
    label "6"
    pos 0.28183784439970383
    pos 0.7558042041572239
    cpu 59
    max_cpu 59
  ]
  node [
    id 7
    label "7"
    pos 0.6183689966753316
    pos 0.25050634136244054
    cpu 69
    max_cpu 69
  ]
  node [
    id 8
    label "8"
    pos 0.9097462559682401
    pos 0.9827854760376531
    cpu 71
    max_cpu 71
  ]
  node [
    id 9
    label "9"
    pos 0.8102172359965896
    pos 0.9021659504395827
    cpu 100
    max_cpu 100
  ]
  node [
    id 10
    label "10"
    pos 0.3101475693193326
    pos 0.7298317482601286
    cpu 86
    max_cpu 86
  ]
  node [
    id 11
    label "11"
    pos 0.8988382879679935
    pos 0.6839839319154413
    cpu 73
    max_cpu 73
  ]
  node [
    id 12
    label "12"
    pos 0.47214271545271336
    pos 0.1007012080683658
    cpu 56
    max_cpu 56
  ]
  node [
    id 13
    label "13"
    pos 0.4341718354537837
    pos 0.6108869734438016
    cpu 74
    max_cpu 74
  ]
  node [
    id 14
    label "14"
    pos 0.9130110532378982
    pos 0.9666063677707588
    cpu 74
    max_cpu 74
  ]
  node [
    id 15
    label "15"
    pos 0.47700977655271704
    pos 0.8653099277716401
    cpu 62
    max_cpu 62
  ]
  node [
    id 16
    label "16"
    pos 0.2604923103919594
    pos 0.8050278270130223
    cpu 51
    max_cpu 51
  ]
  node [
    id 17
    label "17"
    pos 0.5486993038355893
    pos 0.014041700164018955
    cpu 88
    max_cpu 88
  ]
  node [
    id 18
    label "18"
    pos 0.7197046864039541
    pos 0.39882354222426875
    cpu 89
    max_cpu 89
  ]
  node [
    id 19
    label "19"
    pos 0.824844977148233
    pos 0.6681532012318508
    cpu 73
    max_cpu 73
  ]
  node [
    id 20
    label "20"
    pos 0.0011428193144282783
    pos 0.49357786646532464
    cpu 96
    max_cpu 96
  ]
  node [
    id 21
    label "21"
    pos 0.8676027754927809
    pos 0.24391087688713198
    cpu 74
    max_cpu 74
  ]
  node [
    id 22
    label "22"
    pos 0.32520436274739006
    pos 0.8704712321086546
    cpu 67
    max_cpu 67
  ]
  node [
    id 23
    label "23"
    pos 0.19106709150239054
    pos 0.5675107406206719
    cpu 87
    max_cpu 87
  ]
  node [
    id 24
    label "24"
    pos 0.23861592861522019
    pos 0.9675402502901433
    cpu 75
    max_cpu 75
  ]
  node [
    id 25
    label "25"
    pos 0.80317946927987
    pos 0.44796957143557037
    cpu 63
    max_cpu 63
  ]
  node [
    id 26
    label "26"
    pos 0.08044581855253541
    pos 0.32005460467254576
    cpu 58
    max_cpu 58
  ]
  node [
    id 27
    label "27"
    pos 0.5079406425205739
    pos 0.9328338242269067
    cpu 59
    max_cpu 59
  ]
  node [
    id 28
    label "28"
    pos 0.10905784593110368
    pos 0.5512672460905512
    cpu 70
    max_cpu 70
  ]
  node [
    id 29
    label "29"
    pos 0.7065614098668896
    pos 0.5474409113284238
    cpu 66
    max_cpu 66
  ]
  node [
    id 30
    label "30"
    pos 0.814466863291336
    pos 0.540283606970324
    cpu 55
    max_cpu 55
  ]
  node [
    id 31
    label "31"
    pos 0.9638385459738009
    pos 0.603185627961383
    cpu 65
    max_cpu 65
  ]
  node [
    id 32
    label "32"
    pos 0.5876170641754364
    pos 0.4449890262755162
    cpu 97
    max_cpu 97
  ]
  node [
    id 33
    label "33"
    pos 0.5962868615831063
    pos 0.38490114597266045
    cpu 50
    max_cpu 50
  ]
  node [
    id 34
    label "34"
    pos 0.5756510141648885
    pos 0.290329502402758
    cpu 68
    max_cpu 68
  ]
  node [
    id 35
    label "35"
    pos 0.18939132855435614
    pos 0.1867295282555551
    cpu 85
    max_cpu 85
  ]
  node [
    id 36
    label "36"
    pos 0.6127731798686067
    pos 0.6566593889896288
    cpu 74
    max_cpu 74
  ]
  node [
    id 37
    label "37"
    pos 0.47653099200938076
    pos 0.08982436119559367
    cpu 99
    max_cpu 99
  ]
  node [
    id 38
    label "38"
    pos 0.7576039219664368
    pos 0.8767703708227748
    cpu 79
    max_cpu 79
  ]
  node [
    id 39
    label "39"
    pos 0.9233810159462806
    pos 0.8424602231401824
    cpu 69
    max_cpu 69
  ]
  node [
    id 40
    label "40"
    pos 0.898173121357879
    pos 0.9230824398201768
    cpu 69
    max_cpu 69
  ]
  node [
    id 41
    label "41"
    pos 0.5405999249480544
    pos 0.3912960502346249
    cpu 64
    max_cpu 64
  ]
  node [
    id 42
    label "42"
    pos 0.7052833998544062
    pos 0.27563412131212717
    cpu 89
    max_cpu 89
  ]
  node [
    id 43
    label "43"
    pos 0.8116287085078785
    pos 0.8494859651863671
    cpu 82
    max_cpu 82
  ]
  node [
    id 44
    label "44"
    pos 0.8950389674266752
    pos 0.5898011835311598
    cpu 51
    max_cpu 51
  ]
  node [
    id 45
    label "45"
    pos 0.9497648732321206
    pos 0.5796950107456059
    cpu 59
    max_cpu 59
  ]
  node [
    id 46
    label "46"
    pos 0.4505631066311552
    pos 0.660245378622389
    cpu 82
    max_cpu 82
  ]
  node [
    id 47
    label "47"
    pos 0.9962578393535727
    pos 0.9169412179474561
    cpu 81
    max_cpu 81
  ]
  node [
    id 48
    label "48"
    pos 0.7933250841302242
    pos 0.0823729881966474
    cpu 60
    max_cpu 60
  ]
  node [
    id 49
    label "49"
    pos 0.6127831050407122
    pos 0.4864442019691668
    cpu 73
    max_cpu 73
  ]
  node [
    id 50
    label "50"
    pos 0.6301473404114728
    pos 0.8450775756715152
    cpu 85
    max_cpu 85
  ]
  node [
    id 51
    label "51"
    pos 0.24303562206185625
    pos 0.7314892207908478
    cpu 61
    max_cpu 61
  ]
  node [
    id 52
    label "52"
    pos 0.11713429320851798
    pos 0.22046053686782852
    cpu 100
    max_cpu 100
  ]
  node [
    id 53
    label "53"
    pos 0.7945829717105759
    pos 0.33253614921965546
    cpu 78
    max_cpu 78
  ]
  node [
    id 54
    label "54"
    pos 0.8159130965336595
    pos 0.1006075202160962
    cpu 84
    max_cpu 84
  ]
  node [
    id 55
    label "55"
    pos 0.14635848891230385
    pos 0.6976706401912388
    cpu 50
    max_cpu 50
  ]
  node [
    id 56
    label "56"
    pos 0.04523406786561235
    pos 0.5738660367891669
    cpu 50
    max_cpu 50
  ]
  node [
    id 57
    label "57"
    pos 0.9100160146990397
    pos 0.534197968260724
    cpu 86
    max_cpu 86
  ]
  node [
    id 58
    label "58"
    pos 0.6805891325622565
    pos 0.026696794662205203
    cpu 55
    max_cpu 55
  ]
  node [
    id 59
    label "59"
    pos 0.6349999099114583
    pos 0.6063384177542189
    cpu 88
    max_cpu 88
  ]
  node [
    id 60
    label "60"
    pos 0.5759529480315407
    pos 0.3912094093228269
    cpu 90
    max_cpu 90
  ]
  node [
    id 61
    label "61"
    pos 0.3701399403351875
    pos 0.9805166506472687
    cpu 67
    max_cpu 67
  ]
  node [
    id 62
    label "62"
    pos 0.036392037611485795
    pos 0.021636509855024078
    cpu 65
    max_cpu 65
  ]
  node [
    id 63
    label "63"
    pos 0.9610312802396112
    pos 0.18497194139743833
    cpu 54
    max_cpu 54
  ]
  node [
    id 64
    label "64"
    pos 0.12389516442443171
    pos 0.21057650988664645
    cpu 91
    max_cpu 91
  ]
  node [
    id 65
    label "65"
    pos 0.8007465903541809
    pos 0.9369691586445807
    cpu 92
    max_cpu 92
  ]
  node [
    id 66
    label "66"
    pos 0.022782575668658378
    pos 0.42561883196681716
    cpu 81
    max_cpu 81
  ]
  node [
    id 67
    label "67"
    pos 0.10150021937416975
    pos 0.259919889792832
    cpu 51
    max_cpu 51
  ]
  node [
    id 68
    label "68"
    pos 0.22082927131631735
    pos 0.6469257198353225
    cpu 51
    max_cpu 51
  ]
  node [
    id 69
    label "69"
    pos 0.3502939673965323
    pos 0.18031790152968785
    cpu 89
    max_cpu 89
  ]
  node [
    id 70
    label "70"
    pos 0.5036365052098872
    pos 0.03937870708469238
    cpu 91
    max_cpu 91
  ]
  node [
    id 71
    label "71"
    pos 0.10092124118896661
    pos 0.9882351487225011
    cpu 85
    max_cpu 85
  ]
  node [
    id 72
    label "72"
    pos 0.19935579046706298
    pos 0.35855530131160185
    cpu 88
    max_cpu 88
  ]
  node [
    id 73
    label "73"
    pos 0.7315983062253606
    pos 0.8383265651934163
    cpu 61
    max_cpu 61
  ]
  node [
    id 74
    label "74"
    pos 0.9184820619953314
    pos 0.16942460609746768
    cpu 96
    max_cpu 96
  ]
  node [
    id 75
    label "75"
    pos 0.6726405635730526
    pos 0.9665489030431832
    cpu 68
    max_cpu 68
  ]
  node [
    id 76
    label "76"
    pos 0.05805094382649867
    pos 0.6762017842993783
    cpu 77
    max_cpu 77
  ]
  node [
    id 77
    label "77"
    pos 0.8454245937016164
    pos 0.342312541078584
    cpu 50
    max_cpu 50
  ]
  node [
    id 78
    label "78"
    pos 0.25068733928511167
    pos 0.596791393469411
    cpu 64
    max_cpu 64
  ]
  node [
    id 79
    label "79"
    pos 0.44231403369907896
    pos 0.17481948445144113
    cpu 85
    max_cpu 85
  ]
  node [
    id 80
    label "80"
    pos 0.47162541509628797
    pos 0.40990539565755457
    cpu 62
    max_cpu 62
  ]
  node [
    id 81
    label "81"
    pos 0.5691127395242802
    pos 0.5086001300626332
    cpu 92
    max_cpu 92
  ]
  node [
    id 82
    label "82"
    pos 0.3114460010002068
    pos 0.35715168259026286
    cpu 70
    max_cpu 70
  ]
  node [
    id 83
    label "83"
    pos 0.837661174368979
    pos 0.25093266482213705
    cpu 61
    max_cpu 61
  ]
  node [
    id 84
    label "84"
    pos 0.560600218853524
    pos 0.012436318829314397
    cpu 54
    max_cpu 54
  ]
  node [
    id 85
    label "85"
    pos 0.7415743774106636
    pos 0.3359165544734606
    cpu 56
    max_cpu 56
  ]
  node [
    id 86
    label "86"
    pos 0.04569649356841665
    pos 0.28088316421834825
    cpu 54
    max_cpu 54
  ]
  node [
    id 87
    label "87"
    pos 0.24013040782635398
    pos 0.9531293398277989
    cpu 97
    max_cpu 97
  ]
  node [
    id 88
    label "88"
    pos 0.35222556151550743
    pos 0.2878779148564
    cpu 53
    max_cpu 53
  ]
  node [
    id 89
    label "89"
    pos 0.35920119725374633
    pos 0.9469058356578911
    cpu 62
    max_cpu 62
  ]
  node [
    id 90
    label "90"
    pos 0.6337478522492526
    pos 0.6210768456186673
    cpu 86
    max_cpu 86
  ]
  node [
    id 91
    label "91"
    pos 0.7156193503014563
    pos 0.38801723531250565
    cpu 90
    max_cpu 90
  ]
  node [
    id 92
    label "92"
    pos 0.4144179882772473
    pos 0.650832862263345
    cpu 64
    max_cpu 64
  ]
  node [
    id 93
    label "93"
    pos 0.001524221856720187
    pos 0.1923095412446758
    cpu 65
    max_cpu 65
  ]
  node [
    id 94
    label "94"
    pos 0.3344016906625016
    pos 0.23941596018595857
    cpu 70
    max_cpu 70
  ]
  node [
    id 95
    label "95"
    pos 0.6373994011293003
    pos 0.37864807032309444
    cpu 85
    max_cpu 85
  ]
  node [
    id 96
    label "96"
    pos 0.8754233917130172
    pos 0.5681514209101919
    cpu 73
    max_cpu 73
  ]
  node [
    id 97
    label "97"
    pos 0.4144063966836443
    pos 0.40226707511907955
    cpu 65
    max_cpu 65
  ]
  node [
    id 98
    label "98"
    pos 0.7018296239336754
    pos 0.41822655329246605
    cpu 63
    max_cpu 63
  ]
  node [
    id 99
    label "99"
    pos 0.6621958889738174
    pos 0.04677968595679827
    cpu 71
    max_cpu 71
  ]
  edge [
    source 0
    target 18
    bw 98
    max_bw 98
    ltc 0.3801701545288392
  ]
  edge [
    source 0
    target 26
    bw 99
    max_bw 99
    ltc 0.8805768633568337
  ]
  edge [
    source 0
    target 29
    bw 55
    max_bw 55
    ltc 0.2516375002355711
  ]
  edge [
    source 0
    target 30
    bw 91
    max_bw 91
    ltc 0.21972227183034004
  ]
  edge [
    source 0
    target 38
    bw 85
    max_bw 85
    ltc 0.14715497652710452
  ]
  edge [
    source 0
    target 39
    bw 50
    max_bw 50
    ltc 0.11565372148683184
  ]
  edge [
    source 0
    target 43
    bw 81
    max_bw 81
    ltc 0.09722868463652969
  ]
  edge [
    source 0
    target 49
    bw 55
    max_bw 55
    ltc 0.3568953601606792
  ]
  edge [
    source 0
    target 50
    bw 80
    max_bw 80
    ltc 0.23130934555204302
  ]
  edge [
    source 0
    target 65
    bw 50
    max_bw 50
    ltc 0.18426559960612804
  ]
  edge [
    source 0
    target 67
    bw 99
    max_bw 99
    ltc 0.8944109390005038
  ]
  edge [
    source 0
    target 74
    bw 100
    max_bw 100
    ltc 0.5931713382714737
  ]
  edge [
    source 0
    target 75
    bw 86
    max_bw 86
    ltc 0.2702230122761885
  ]
  edge [
    source 0
    target 83
    bw 84
    max_bw 84
    ltc 0.5070668098781206
  ]
  edge [
    source 0
    target 90
    bw 98
    max_bw 98
    ltc 0.2512349491396359
  ]
  edge [
    source 1
    target 12
    bw 79
    max_bw 79
    ltc 0.16640835233728588
  ]
  edge [
    source 1
    target 13
    bw 53
    max_bw 53
    ltc 0.3522328844822125
  ]
  edge [
    source 1
    target 33
    bw 84
    max_bw 84
    ltc 0.21621269122006806
  ]
  edge [
    source 1
    target 36
    bw 92
    max_bw 92
    ltc 0.44174728217619935
  ]
  edge [
    source 1
    target 63
    bw 63
    max_bw 63
    ltc 0.5454947492392659
  ]
  edge [
    source 1
    target 64
    bw 98
    max_bw 98
    ltc 0.3005888801906884
  ]
  edge [
    source 1
    target 72
    bw 89
    max_bw 89
    ltc 0.24261959268650704
  ]
  edge [
    source 1
    target 74
    bw 71
    max_bw 71
    ltc 0.505889010679376
  ]
  edge [
    source 1
    target 76
    bw 59
    max_bw 59
    ltc 0.552763974820874
  ]
  edge [
    source 1
    target 77
    bw 50
    max_bw 50
    ltc 0.4329606684978797
  ]
  edge [
    source 1
    target 79
    bw 60
    max_bw 60
    ltc 0.08686244515754224
  ]
  edge [
    source 1
    target 80
    bw 100
    max_bw 100
    ltc 0.15938652710388546
  ]
  edge [
    source 1
    target 81
    bw 93
    max_bw 93
    ltc 0.2905275648870404
  ]
  edge [
    source 1
    target 94
    bw 73
    max_bw 73
    ltc 0.08834891502685148
  ]
  edge [
    source 2
    target 8
    bw 52
    max_bw 52
    ltc 0.7019200334747892
  ]
  edge [
    source 2
    target 12
    bw 84
    max_bw 84
    ltc 0.30673928539948087
  ]
  edge [
    source 2
    target 18
    bw 85
    max_bw 85
    ltc 0.20851951874741412
  ]
  edge [
    source 2
    target 23
    bw 80
    max_bw 80
    ltc 0.3591156890514814
  ]
  edge [
    source 2
    target 24
    bw 53
    max_bw 53
    ltc 0.6251947340392152
  ]
  edge [
    source 2
    target 35
    bw 68
    max_bw 68
    ltc 0.388872948459385
  ]
  edge [
    source 2
    target 38
    bw 96
    max_bw 96
    ltc 0.5322663864928742
  ]
  edge [
    source 2
    target 41
    bw 85
    max_bw 85
    ltc 0.03234138197234968
  ]
  edge [
    source 2
    target 72
    bw 70
    max_bw 70
    ltc 0.31534808687600113
  ]
  edge [
    source 2
    target 80
    bw 67
    max_bw 67
    ltc 0.0399597409405809
  ]
  edge [
    source 2
    target 96
    bw 100
    max_bw 100
    ltc 0.39905405114293757
  ]
  edge [
    source 2
    target 99
    bw 77
    max_bw 77
    ltc 0.38865384335680064
  ]
  edge [
    source 3
    target 7
    bw 64
    max_bw 64
    ltc 0.17365328760215407
  ]
  edge [
    source 3
    target 10
    bw 91
    max_bw 91
    ltc 0.6373882370736725
  ]
  edge [
    source 3
    target 12
    bw 51
    max_bw 51
    ltc 0.37172679587129664
  ]
  edge [
    source 3
    target 18
    bw 86
    max_bw 86
    ltc 0.11502323397991394
  ]
  edge [
    source 3
    target 21
    bw 60
    max_bw 60
    ltc 0.10272157200555988
  ]
  edge [
    source 3
    target 22
    bw 72
    max_bw 72
    ltc 0.729367832678406
  ]
  edge [
    source 3
    target 31
    bw 93
    max_bw 93
    ltc 0.3497686998260593
  ]
  edge [
    source 3
    target 41
    bw 90
    max_bw 90
    ltc 0.2586245455154072
  ]
  edge [
    source 3
    target 53
    bw 61
    max_bw 61
    ltc 0.031149821343953214
  ]
  edge [
    source 3
    target 54
    bw 52
    max_bw 52
    ltc 0.20523338440855296
  ]
  edge [
    source 3
    target 60
    bw 66
    max_bw 66
    ltc 0.22566709421030515
  ]
  edge [
    source 3
    target 63
    bw 82
    max_bw 82
    ltc 0.2131102253547216
  ]
  edge [
    source 3
    target 77
    bw 50
    max_bw 50
    ltc 0.07292976087445567
  ]
  edge [
    source 3
    target 85
    bw 88
    max_bw 88
    ltc 0.05334691812333409
  ]
  edge [
    source 3
    target 93
    bw 69
    max_bw 69
    ltc 0.7901106837571362
  ]
  edge [
    source 3
    target 96
    bw 96
    max_bw 96
    ltc 0.28024032319734943
  ]
  edge [
    source 4
    target 6
    bw 92
    max_bw 92
    ltc 0.26011634649178744
  ]
  edge [
    source 4
    target 13
    bw 90
    max_bw 90
    ltc 0.05056097398502675
  ]
  edge [
    source 4
    target 15
    bw 63
    max_bw 63
    ltc 0.28192819056099466
  ]
  edge [
    source 4
    target 17
    bw 80
    max_bw 80
    ltc 0.5738877684476715
  ]
  edge [
    source 4
    target 22
    bw 74
    max_bw 74
    ltc 0.3245611210093104
  ]
  edge [
    source 4
    target 28
    bw 52
    max_bw 52
    ltc 0.36893950184944485
  ]
  edge [
    source 4
    target 50
    bw 53
    max_bw 53
    ltc 0.3034176573569131
  ]
  edge [
    source 4
    target 66
    bw 80
    max_bw 80
    ltc 0.480454701044244
  ]
  edge [
    source 4
    target 68
    bw 84
    max_bw 84
    ltc 0.26354298871263676
  ]
  edge [
    source 4
    target 71
    bw 93
    max_bw 93
    ltc 0.5523027081176488
  ]
  edge [
    source 4
    target 73
    bw 63
    max_bw 63
    ltc 0.36058619047743723
  ]
  edge [
    source 4
    target 78
    bw 98
    max_bw 98
    ltc 0.22630723555500726
  ]
  edge [
    source 4
    target 79
    bw 90
    max_bw 90
    ltc 0.4099983902235072
  ]
  edge [
    source 4
    target 88
    bw 58
    max_bw 58
    ltc 0.3206102477483096
  ]
  edge [
    source 4
    target 98
    bw 69
    max_bw 69
    ltc 0.27929570377364243
  ]
  edge [
    source 5
    target 11
    bw 81
    max_bw 81
    ltc 0.1795367919147485
  ]
  edge [
    source 5
    target 15
    bw 58
    max_bw 58
    ltc 0.5620488326711915
  ]
  edge [
    source 5
    target 18
    bw 76
    max_bw 76
    ltc 0.2161126801849631
  ]
  edge [
    source 5
    target 19
    bw 52
    max_bw 52
    ltc 0.18345242051742025
  ]
  edge [
    source 5
    target 42
    bw 53
    max_bw 53
    ltc 0.30594926917383647
  ]
  edge [
    source 5
    target 44
    bw 94
    max_bw 94
    ltc 0.08611257810557438
  ]
  edge [
    source 5
    target 54
    bw 64
    max_bw 64
    ltc 0.4144646070405058
  ]
  edge [
    source 5
    target 57
    bw 82
    max_bw 82
    ltc 0.02957241382692587
  ]
  edge [
    source 5
    target 60
    bw 54
    max_bw 54
    ltc 0.3510090521903707
  ]
  edge [
    source 5
    target 74
    bw 53
    max_bw 53
    ltc 0.3354225632165691
  ]
  edge [
    source 5
    target 79
    bw 95
    max_bw 95
    ltc 0.5707723300468518
  ]
  edge [
    source 5
    target 83
    bw 61
    max_bw 61
    ltc 0.2633526779928694
  ]
  edge [
    source 5
    target 85
    bw 72
    max_bw 72
    ltc 0.23710438458802477
  ]
  edge [
    source 5
    target 90
    bw 63
    max_bw 63
    ltc 0.2980315436819505
  ]
  edge [
    source 5
    target 94
    bw 95
    max_bw 95
    ltc 0.632070552075789
  ]
  edge [
    source 6
    target 10
    bw 61
    max_bw 61
    ltc 0.038418862429171585
  ]
  edge [
    source 6
    target 46
    bw 66
    max_bw 66
    ltc 0.19390642911638864
  ]
  edge [
    source 6
    target 51
    bw 74
    max_bw 74
    ltc 0.04579116589979892
  ]
  edge [
    source 6
    target 56
    bw 79
    max_bw 79
    ltc 0.2984674920580823
  ]
  edge [
    source 6
    target 75
    bw 71
    max_bw 71
    ltc 0.444004609685324
  ]
  edge [
    source 6
    target 90
    bw 96
    max_bw 96
    ltc 0.37681841086051815
  ]
  edge [
    source 6
    target 91
    bw 75
    max_bw 75
    ltc 0.5687122728712454
  ]
  edge [
    source 7
    target 12
    bw 66
    max_bw 66
    ltc 0.20934111703497654
  ]
  edge [
    source 7
    target 13
    bw 69
    max_bw 69
    ltc 0.4047255788573762
  ]
  edge [
    source 7
    target 15
    bw 83
    max_bw 83
    ltc 0.6308453685137808
  ]
  edge [
    source 7
    target 19
    bw 90
    max_bw 90
    ltc 0.46589830443029606
  ]
  edge [
    source 7
    target 33
    bw 82
    max_bw 82
    ltc 0.13619685824735633
  ]
  edge [
    source 7
    target 38
    bw 86
    max_bw 86
    ltc 0.6415551410570365
  ]
  edge [
    source 7
    target 42
    bw 56
    max_bw 56
    ltc 0.09047385702607472
  ]
  edge [
    source 7
    target 52
    bw 71
    max_bw 71
    ltc 0.5021344225674964
  ]
  edge [
    source 7
    target 54
    bw 81
    max_bw 81
    ltc 0.24797848287684307
  ]
  edge [
    source 7
    target 58
    bw 63
    max_bw 63
    ltc 0.23229734932614324
  ]
  edge [
    source 7
    target 68
    bw 57
    max_bw 57
    ltc 0.5614144252398102
  ]
  edge [
    source 7
    target 70
    bw 74
    max_bw 74
    ltc 0.2402882072711577
  ]
  edge [
    source 7
    target 93
    bw 65
    max_bw 65
    ltc 0.6195840086420651
  ]
  edge [
    source 7
    target 95
    bw 91
    max_bw 91
    ltc 0.12954713039935434
  ]
  edge [
    source 8
    target 28
    bw 68
    max_bw 68
    ltc 0.9095657825272878
  ]
  edge [
    source 8
    target 38
    bw 90
    max_bw 90
    ltc 0.18543595209461947
  ]
  edge [
    source 8
    target 39
    bw 65
    max_bw 65
    ltc 0.1409861102392639
  ]
  edge [
    source 8
    target 40
    bw 61
    max_bw 61
    ltc 0.06081438956607906
  ]
  edge [
    source 8
    target 87
    bw 88
    max_bw 88
    ltc 0.6702722361083405
  ]
  edge [
    source 9
    target 27
    bw 97
    max_bw 97
    ltc 0.3038283354891371
  ]
  edge [
    source 9
    target 29
    bw 79
    max_bw 79
    ltc 0.36955971596352843
  ]
  edge [
    source 9
    target 40
    bw 51
    max_bw 51
    ltc 0.0904087235707749
  ]
  edge [
    source 9
    target 43
    bw 81
    max_bw 81
    ltc 0.05269889088898492
  ]
  edge [
    source 9
    target 46
    bw 94
    max_bw 94
    ltc 0.43344740838764584
  ]
  edge [
    source 9
    target 49
    bw 74
    max_bw 74
    ltc 0.46022256378582993
  ]
  edge [
    source 9
    target 50
    bw 74
    max_bw 74
    ltc 0.18890275230841447
  ]
  edge [
    source 9
    target 53
    bw 53
    max_bw 53
    ltc 0.5698443126483046
  ]
  edge [
    source 9
    target 65
    bw 68
    max_bw 68
    ltc 0.03606877361714031
  ]
  edge [
    source 9
    target 81
    bw 97
    max_bw 97
    ltc 0.46154678331470056
  ]
  edge [
    source 9
    target 97
    bw 53
    max_bw 53
    ltc 0.6376245808187778
  ]
  edge [
    source 10
    target 16
    bw 92
    max_bw 92
    ltc 0.09011156972861957
  ]
  edge [
    source 10
    target 35
    bw 62
    max_bw 62
    ltc 0.5563650699474089
  ]
  edge [
    source 10
    target 37
    bw 88
    max_bw 88
    ltc 0.6612812554754812
  ]
  edge [
    source 10
    target 41
    bw 85
    max_bw 85
    ltc 0.409529861002152
  ]
  edge [
    source 10
    target 49
    bw 72
    max_bw 72
    ltc 0.38836292970738273
  ]
  edge [
    source 10
    target 50
    bw 55
    max_bw 55
    ltc 0.3401197645458982
  ]
  edge [
    source 10
    target 51
    bw 73
    max_bw 73
    ltc 0.06713241154524675
  ]
  edge [
    source 10
    target 68
    bw 93
    max_bw 93
    ltc 0.12186536795714441
  ]
  edge [
    source 10
    target 73
    bw 82
    max_bw 82
    ltc 0.4351917381339191
  ]
  edge [
    source 10
    target 78
    bw 61
    max_bw 61
    ltc 0.1457232135198867
  ]
  edge [
    source 10
    target 80
    bw 90
    max_bw 90
    ltc 0.3583684776404765
  ]
  edge [
    source 10
    target 85
    bw 70
    max_bw 70
    ltc 0.5842073866666377
  ]
  edge [
    source 10
    target 89
    bw 60
    max_bw 60
    ltc 0.2225475630805678
  ]
  edge [
    source 11
    target 19
    bw 93
    max_bw 93
    ltc 0.07566784046109716
  ]
  edge [
    source 11
    target 25
    bw 87
    max_bw 87
    ltc 0.25466328346569655
  ]
  edge [
    source 11
    target 27
    bw 78
    max_bw 78
    ltc 0.463386704729176
  ]
  edge [
    source 11
    target 29
    bw 97
    max_bw 97
    ltc 0.2358270432400267
  ]
  edge [
    source 11
    target 33
    bw 90
    max_bw 90
    ltc 0.4254267016241753
  ]
  edge [
    source 11
    target 36
    bw 52
    max_bw 52
    ltc 0.28736714620502185
  ]
  edge [
    source 11
    target 38
    bw 77
    max_bw 77
    ltc 0.23898484714816692
  ]
  edge [
    source 11
    target 43
    bw 69
    max_bw 69
    ltc 0.18707333793574313
  ]
  edge [
    source 11
    target 75
    bw 75
    max_bw 75
    ltc 0.36195078868524555
  ]
  edge [
    source 11
    target 77
    bw 73
    max_bw 73
    ltc 0.34582128629044
  ]
  edge [
    source 11
    target 79
    bw 96
    max_bw 96
    ltc 0.6838587787673045
  ]
  edge [
    source 12
    target 37
    bw 70
    max_bw 70
    ltc 0.011728715574670976
  ]
  edge [
    source 12
    target 69
    bw 79
    max_bw 79
    ltc 0.14555389132748328
  ]
  edge [
    source 12
    target 79
    bw 53
    max_bw 53
    ltc 0.07989536375258294
  ]
  edge [
    source 12
    target 84
    bw 85
    max_bw 85
    ltc 0.12496167644636667
  ]
  edge [
    source 12
    target 88
    bw 89
    max_bw 89
    ltc 0.22229539665142253
  ]
  edge [
    source 12
    target 97
    bw 59
    max_bw 59
    ltc 0.3070430827669829
  ]
  edge [
    source 13
    target 23
    bw 59
    max_bw 59
    ltc 0.2469441517946898
  ]
  edge [
    source 13
    target 30
    bw 91
    max_bw 91
    ltc 0.3867934119854588
  ]
  edge [
    source 13
    target 36
    bw 73
    max_bw 73
    ltc 0.18437340982821276
  ]
  edge [
    source 13
    target 41
    bw 53
    max_bw 53
    ltc 0.24402276899760614
  ]
  edge [
    source 13
    target 46
    bw 96
    max_bw 96
    ltc 0.05200890243586889
  ]
  edge [
    source 13
    target 55
    bw 76
    max_bw 76
    ltc 0.3006125866651897
  ]
  edge [
    source 13
    target 60
    bw 100
    max_bw 100
    ltc 0.2614576754694946
  ]
  edge [
    source 13
    target 62
    bw 94
    max_bw 94
    ltc 0.7109464652215726
  ]
  edge [
    source 13
    target 72
    bw 53
    max_bw 53
    ltc 0.34468804409819265
  ]
  edge [
    source 13
    target 73
    bw 81
    max_bw 81
    ltc 0.37442125128104986
  ]
  edge [
    source 13
    target 82
    bw 59
    max_bw 59
    ltc 0.2818567513238891
  ]
  edge [
    source 13
    target 95
    bw 60
    max_bw 60
    ltc 0.3086038748510874
  ]
  edge [
    source 14
    target 45
    bw 77
    max_bw 77
    ltc 0.3886531120153459
  ]
  edge [
    source 14
    target 69
    bw 95
    max_bw 95
    ltc 0.966902306772269
  ]
  edge [
    source 14
    target 73
    bw 57
    max_bw 57
    ltc 0.22218526622599652
  ]
  edge [
    source 15
    target 34
    bw 89
    max_bw 89
    ltc 0.5833803076167736
  ]
  edge [
    source 15
    target 36
    bw 71
    max_bw 71
    ltc 0.2489312134183588
  ]
  edge [
    source 15
    target 50
    bw 83
    max_bw 83
    ltc 0.15446831887509468
  ]
  edge [
    source 15
    target 56
    bw 94
    max_bw 94
    ltc 0.5209316694186054
  ]
  edge [
    source 15
    target 88
    bw 84
    max_bw 84
    ltc 0.590761229145751
  ]
  edge [
    source 15
    target 89
    bw 84
    max_bw 84
    ltc 0.1433065020863444
  ]
  edge [
    source 16
    target 27
    bw 74
    max_bw 74
    ltc 0.2785050268793519
  ]
  edge [
    source 16
    target 36
    bw 83
    max_bw 83
    ltc 0.38224992400355573
  ]
  edge [
    source 16
    target 43
    bw 55
    max_bw 55
    ltc 0.5529266274814671
  ]
  edge [
    source 16
    target 50
    bw 90
    max_bw 90
    ltc 0.37181826688092634
  ]
  edge [
    source 16
    target 53
    bw 86
    max_bw 86
    ltc 0.7130927149338632
  ]
  edge [
    source 16
    target 61
    bw 50
    max_bw 50
    ltc 0.2069273543364886
  ]
  edge [
    source 16
    target 75
    bw 61
    max_bw 61
    ltc 0.4426683189501678
  ]
  edge [
    source 16
    target 76
    bw 84
    max_bw 84
    ltc 0.23995552958443964
  ]
  edge [
    source 16
    target 92
    bw 55
    max_bw 55
    ltc 0.21787427903862272
  ]
  edge [
    source 17
    target 23
    bw 66
    max_bw 66
    ltc 0.6589603766861452
  ]
  edge [
    source 17
    target 60
    bw 58
    max_bw 58
    ltc 0.3781510835024311
  ]
  edge [
    source 17
    target 77
    bw 51
    max_bw 51
    ltc 0.44250157360266357
  ]
  edge [
    source 17
    target 84
    bw 67
    max_bw 67
    ltc 0.012008706320579278
  ]
  edge [
    source 17
    target 93
    bw 98
    max_bw 98
    ltc 0.5754824006884489
  ]
  edge [
    source 18
    target 29
    bw 85
    max_bw 85
    ltc 0.14919741324021138
  ]
  edge [
    source 18
    target 32
    bw 77
    max_bw 77
    ltc 0.13992280680314273
  ]
  edge [
    source 18
    target 33
    bw 90
    max_bw 90
    ltc 0.12420061433381191
  ]
  edge [
    source 18
    target 37
    bw 86
    max_bw 86
    ltc 0.39320979072487156
  ]
  edge [
    source 18
    target 41
    bw 98
    max_bw 98
    ltc 0.17926287600010463
  ]
  edge [
    source 18
    target 42
    bw 75
    max_bw 75
    ltc 0.12403066931373448
  ]
  edge [
    source 18
    target 46
    bw 53
    max_bw 53
    ltc 0.3752046994753177
  ]
  edge [
    source 18
    target 53
    bw 89
    max_bw 89
    ltc 0.10000388033375457
  ]
  edge [
    source 18
    target 56
    bw 85
    max_bw 85
    ltc 0.6968145306858889
  ]
  edge [
    source 18
    target 77
    bw 100
    max_bw 100
    ltc 0.13783681780068913
  ]
  edge [
    source 18
    target 83
    bw 80
    max_bw 80
    ltc 0.18917041172396756
  ]
  edge [
    source 18
    target 84
    bw 79
    max_bw 79
    ltc 0.4178627980538208
  ]
  edge [
    source 18
    target 91
    bw 99
    max_bw 99
    ltc 0.011552758983965308
  ]
  edge [
    source 18
    target 98
    bw 83
    max_bw 83
    ltc 0.0263817114082645
  ]
  edge [
    source 19
    target 31
    bw 68
    max_bw 68
    ltc 0.15342749998459043
  ]
  edge [
    source 19
    target 38
    bw 67
    max_bw 67
    ltc 0.21918595518438144
  ]
  edge [
    source 19
    target 59
    bw 79
    max_bw 79
    ltc 0.19965524538694704
  ]
  edge [
    source 19
    target 74
    bw 70
    max_bw 70
    ltc 0.5074427211650376
  ]
  edge [
    source 19
    target 83
    bw 52
    max_bw 52
    ltc 0.4174173342270423
  ]
  edge [
    source 19
    target 91
    bw 55
    max_bw 55
    ltc 0.3006765653682306
  ]
  edge [
    source 20
    target 28
    bw 87
    max_bw 87
    ltc 0.12236714220419331
  ]
  edge [
    source 20
    target 56
    bw 62
    max_bw 62
    ltc 0.0915981904447495
  ]
  edge [
    source 20
    target 64
    bw 94
    max_bw 94
    ltc 0.308476751239664
  ]
  edge [
    source 20
    target 66
    bw 52
    max_bw 52
    ltc 0.0713211709455178
  ]
  edge [
    source 20
    target 67
    bw 97
    max_bw 97
    ltc 0.2542983637568951
  ]
  edge [
    source 20
    target 72
    bw 77
    max_bw 77
    ltc 0.239832180980465
  ]
  edge [
    source 20
    target 78
    bw 71
    max_bw 71
    ltc 0.2700472173602647
  ]
  edge [
    source 20
    target 92
    bw 89
    max_bw 89
    ltc 0.44218265341901647
  ]
  edge [
    source 20
    target 93
    bw 89
    max_bw 89
    ltc 0.30126856664636276
  ]
  edge [
    source 21
    target 31
    bw 61
    max_bw 61
    ltc 0.37194041226995034
  ]
  edge [
    source 21
    target 45
    bw 72
    max_bw 72
    ltc 0.3456900271283717
  ]
  edge [
    source 21
    target 53
    bw 80
    max_bw 80
    ltc 0.11483174926998921
  ]
  edge [
    source 21
    target 57
    bw 67
    max_bw 67
    ltc 0.2933691842680609
  ]
  edge [
    source 21
    target 58
    bw 56
    max_bw 56
    ltc 0.2866287845959699
  ]
  edge [
    source 21
    target 70
    bw 57
    max_bw 57
    ltc 0.4174983286047195
  ]
  edge [
    source 21
    target 77
    bw 68
    max_bw 68
    ltc 0.10087001171413251
  ]
  edge [
    source 21
    target 83
    bw 78
    max_bw 78
    ltc 0.03075394257101087
  ]
  edge [
    source 22
    target 28
    bw 93
    max_bw 93
    ltc 0.38550032609799273
  ]
  edge [
    source 22
    target 41
    bw 69
    max_bw 69
    ltc 0.5253609265445516
  ]
  edge [
    source 22
    target 49
    bw 99
    max_bw 99
    ltc 0.47976899951618807
  ]
  edge [
    source 22
    target 51
    bw 91
    max_bw 91
    ltc 0.16145495166079452
  ]
  edge [
    source 22
    target 59
    bw 79
    max_bw 79
    ltc 0.40711107165180266
  ]
  edge [
    source 23
    target 68
    bw 96
    max_bw 96
    ltc 0.08480876293721017
  ]
  edge [
    source 23
    target 77
    bw 71
    max_bw 71
    ltc 0.6920245441900221
  ]
  edge [
    source 23
    target 81
    bw 59
    max_bw 59
    ltc 0.3826081180850864
  ]
  edge [
    source 24
    target 32
    bw 75
    max_bw 75
    ltc 0.6283801193079754
  ]
  edge [
    source 24
    target 44
    bw 100
    max_bw 100
    ltc 0.7573493305195514
  ]
  edge [
    source 24
    target 48
    bw 82
    max_bw 82
    ltc 1.0446163549812268
  ]
  edge [
    source 24
    target 51
    bw 77
    max_bw 77
    ltc 0.23609240186384556
  ]
  edge [
    source 24
    target 61
    bw 59
    max_bw 59
    ltc 0.1321625991918383
  ]
  edge [
    source 24
    target 68
    bw 78
    max_bw 78
    ltc 0.3211075245406651
  ]
  edge [
    source 25
    target 29
    bw 93
    max_bw 93
    ltc 0.13867082196630182
  ]
  edge [
    source 25
    target 32
    bw 67
    max_bw 67
    ltc 0.2155830098682619
  ]
  edge [
    source 25
    target 33
    bw 91
    max_bw 91
    ltc 0.21629188012946213
  ]
  edge [
    source 25
    target 41
    bw 50
    max_bw 50
    ltc 0.2686259576191734
  ]
  edge [
    source 25
    target 43
    bw 72
    max_bw 72
    ltc 0.40160528394703293
  ]
  edge [
    source 25
    target 44
    bw 66
    max_bw 66
    ltc 0.16898039409770788
  ]
  edge [
    source 25
    target 45
    bw 92
    max_bw 92
    ltc 0.19707580270866923
  ]
  edge [
    source 25
    target 55
    bw 86
    max_bw 86
    ltc 0.7026837297025837
  ]
  edge [
    source 25
    target 64
    bw 80
    max_bw 80
    ltc 0.7195711448456694
  ]
  edge [
    source 25
    target 81
    bw 74
    max_bw 74
    ltc 0.2417918497755994
  ]
  edge [
    source 25
    target 84
    bw 53
    max_bw 53
    ltc 0.498531751108385
  ]
  edge [
    source 25
    target 85
    bw 58
    max_bw 58
    ltc 0.12787128666954217
  ]
  edge [
    source 25
    target 91
    bw 77
    max_bw 77
    ltc 0.10611812777337765
  ]
  edge [
    source 25
    target 95
    bw 79
    max_bw 79
    ltc 0.17969001506064425
  ]
  edge [
    source 26
    target 28
    bw 96
    max_bw 96
    ltc 0.2329762512836944
  ]
  edge [
    source 26
    target 35
    bw 73
    max_bw 73
    ltc 0.17217636350891263
  ]
  edge [
    source 26
    target 52
    bw 82
    max_bw 82
    ltc 0.10613681036508159
  ]
  edge [
    source 26
    target 71
    bw 69
    max_bw 69
    ltc 0.6684941902358126
  ]
  edge [
    source 26
    target 72
    bw 58
    max_bw 58
    ltc 0.12498753962858201
  ]
  edge [
    source 26
    target 81
    bw 57
    max_bw 57
    ltc 0.5237793187943082
  ]
  edge [
    source 26
    target 95
    bw 73
    max_bw 73
    ltc 0.5600272201974072
  ]
  edge [
    source 27
    target 61
    bw 63
    max_bw 63
    ltc 0.145817301642226
  ]
  edge [
    source 27
    target 62
    bw 67
    max_bw 67
    ltc 1.0259817895607304
  ]
  edge [
    source 27
    target 73
    bw 50
    max_bw 50
    ltc 0.24280521523210064
  ]
  edge [
    source 27
    target 78
    bw 61
    max_bw 61
    ltc 0.4232065421221215
  ]
  edge [
    source 28
    target 35
    bw 78
    max_bw 78
    ltc 0.37328436366224566
  ]
  edge [
    source 28
    target 42
    bw 86
    max_bw 86
    ltc 0.656855030144603
  ]
  edge [
    source 28
    target 55
    bw 75
    max_bw 75
    ltc 0.15108041491540922
  ]
  edge [
    source 28
    target 68
    bw 82
    max_bw 82
    ltc 0.1471169437278436
  ]
  edge [
    source 28
    target 75
    bw 92
    max_bw 92
    ltc 0.7000602361411539
  ]
  edge [
    source 28
    target 76
    bw 64
    max_bw 64
    ltc 0.13494570352464588
  ]
  edge [
    source 29
    target 30
    bw 72
    max_bw 72
    ltc 0.10814256277899945
  ]
  edge [
    source 29
    target 34
    bw 78
    max_bw 78
    ltc 0.28852003102487606
  ]
  edge [
    source 29
    target 39
    bw 70
    max_bw 70
    ltc 0.3661244814572032
  ]
  edge [
    source 29
    target 48
    bw 68
    max_bw 68
    ltc 0.4730920716919465
  ]
  edge [
    source 29
    target 53
    bw 54
    max_bw 54
    ltc 0.23223232360381743
  ]
  edge [
    source 29
    target 63
    bw 72
    max_bw 72
    ltc 0.44287545550678165
  ]
  edge [
    source 29
    target 64
    bw 85
    max_bw 85
    ltc 0.6730360900700181
  ]
  edge [
    source 29
    target 68
    bw 69
    max_bw 69
    ltc 0.49581542689247443
  ]
  edge [
    source 29
    target 77
    bw 57
    max_bw 57
    ltc 0.24771078318489176
  ]
  edge [
    source 29
    target 90
    bw 58
    max_bw 58
    ltc 0.10355706152519979
  ]
  edge [
    source 30
    target 36
    bw 63
    max_bw 63
    ltc 0.23285975301290884
  ]
  edge [
    source 30
    target 37
    bw 55
    max_bw 55
    ltc 0.5631289241399637
  ]
  edge [
    source 30
    target 40
    bw 50
    max_bw 50
    ltc 0.39184395372483705
  ]
  edge [
    source 30
    target 45
    bw 58
    max_bw 58
    ltc 0.14092129094453737
  ]
  edge [
    source 30
    target 48
    bw 65
    max_bw 65
    ltc 0.4583984179856957
  ]
  edge [
    source 30
    target 49
    bw 65
    max_bw 65
    ltc 0.20874630505226838
  ]
  edge [
    source 30
    target 51
    bw 61
    max_bw 61
    ltc 0.6025721950186111
  ]
  edge [
    source 30
    target 53
    bw 54
    max_bw 54
    ltc 0.20869685035060115
  ]
  edge [
    source 30
    target 65
    bw 89
    max_bw 89
    ltc 0.39692275419353257
  ]
  edge [
    source 30
    target 75
    bw 78
    max_bw 78
    ltc 0.4492402496747845
  ]
  edge [
    source 30
    target 85
    bw 95
    max_bw 95
    ltc 0.21697743349048937
  ]
  edge [
    source 30
    target 90
    bw 76
    max_bw 76
    ltc 0.1979568345961231
  ]
  edge [
    source 30
    target 91
    bw 60
    max_bw 60
    ltc 0.18153754091680677
  ]
  edge [
    source 30
    target 95
    bw 96
    max_bw 96
    ltc 0.2397476441255301
  ]
  edge [
    source 30
    target 96
    bw 99
    max_bw 99
    ltc 0.06702472238667107
  ]
  edge [
    source 31
    target 32
    bw 77
    max_bw 77
    ltc 0.40812837214718845
  ]
  edge [
    source 31
    target 33
    bw 85
    max_bw 85
    ltc 0.42748374913617676
  ]
  edge [
    source 31
    target 39
    bw 91
    max_bw 91
    ltc 0.24267085452090464
  ]
  edge [
    source 31
    target 45
    bw 68
    max_bw 68
    ltc 0.027383888723446097
  ]
  edge [
    source 31
    target 49
    bw 84
    max_bw 84
    ltc 0.36995740721253495
  ]
  edge [
    source 31
    target 71
    bw 80
    max_bw 80
    ltc 0.9449282556551526
  ]
  edge [
    source 31
    target 77
    bw 93
    max_bw 93
    ltc 0.2864901945136685
  ]
  edge [
    source 31
    target 96
    bw 68
    max_bw 68
    ltc 0.09510328683417787
  ]
  edge [
    source 32
    target 36
    bw 95
    max_bw 95
    ltc 0.21315996952592245
  ]
  edge [
    source 32
    target 41
    bw 73
    max_bw 73
    ltc 0.0713690903490772
  ]
  edge [
    source 32
    target 44
    bw 51
    max_bw 51
    ltc 0.3398216995538209
  ]
  edge [
    source 32
    target 49
    bw 56
    max_bw 56
    ltc 0.04849599163461015
  ]
  edge [
    source 32
    target 60
    bw 92
    max_bw 92
    ltc 0.055029980964891055
  ]
  edge [
    source 32
    target 68
    bw 98
    max_bw 98
    ltc 0.41870241603832564
  ]
  edge [
    source 32
    target 87
    bw 80
    max_bw 80
    ltc 0.6155920358465317
  ]
  edge [
    source 32
    target 99
    bw 66
    max_bw 66
    ltc 0.4051329162452443
  ]
  edge [
    source 33
    target 46
    bw 76
    max_bw 76
    ltc 0.31152826390355676
  ]
  edge [
    source 33
    target 49
    bw 85
    max_bw 85
    ltc 0.10287428380952275
  ]
  edge [
    source 33
    target 53
    bw 99
    max_bw 99
    ltc 0.20509373509843684
  ]
  edge [
    source 33
    target 66
    bw 92
    max_bw 92
    ltc 0.5749479071315587
  ]
  edge [
    source 33
    target 79
    bw 59
    max_bw 59
    ltc 0.2604648464459656
  ]
  edge [
    source 33
    target 91
    bw 94
    max_bw 94
    ltc 0.1193731664843026
  ]
  edge [
    source 33
    target 94
    bw 63
    max_bw 63
    ltc 0.2995826797922495
  ]
  edge [
    source 33
    target 95
    bw 56
    max_bw 56
    ltc 0.041585356353127
  ]
  edge [
    source 33
    target 98
    bw 88
    max_bw 88
    ltc 0.11067907416318437
  ]
  edge [
    source 34
    target 54
    bw 89
    max_bw 89
    ltc 0.3061377120660147
  ]
  edge [
    source 34
    target 57
    bw 58
    max_bw 58
    ltc 0.41384995133758423
  ]
  edge [
    source 34
    target 60
    bw 63
    max_bw 63
    ltc 0.10088035876354522
  ]
  edge [
    source 34
    target 64
    bw 57
    max_bw 57
    ltc 0.4587416348992068
  ]
  edge [
    source 34
    target 72
    bw 66
    max_bw 66
    ltc 0.3824301962640327
  ]
  edge [
    source 34
    target 80
    bw 72
    max_bw 72
    ltc 0.1584920171783542
  ]
  edge [
    source 34
    target 91
    bw 65
    max_bw 65
    ltc 0.17068693061240928
  ]
  edge [
    source 34
    target 97
    bw 75
    max_bw 75
    ltc 0.19629021079079448
  ]
  edge [
    source 35
    target 64
    bw 58
    max_bw 58
    ltc 0.06970241063727715
  ]
  edge [
    source 35
    target 69
    bw 85
    max_bw 85
    ltc 0.16103033299241368
  ]
  edge [
    source 35
    target 70
    bw 56
    max_bw 56
    ltc 0.34707678624620447
  ]
  edge [
    source 36
    target 41
    bw 67
    max_bw 67
    ltc 0.2750030550394486
  ]
  edge [
    source 36
    target 45
    bw 57
    max_bw 57
    ltc 0.34566879655892807
  ]
  edge [
    source 36
    target 50
    bw 70
    max_bw 70
    ltc 0.1892175322930184
  ]
  edge [
    source 36
    target 71
    bw 75
    max_bw 75
    ltc 0.6098646502073534
  ]
  edge [
    source 36
    target 85
    bw 52
    max_bw 52
    ltc 0.3456381263427605
  ]
  edge [
    source 36
    target 88
    bw 66
    max_bw 66
    ltc 0.45153608614744134
  ]
  edge [
    source 36
    target 90
    bw 100
    max_bw 100
    ltc 0.04130440986410276
  ]
  edge [
    source 36
    target 91
    bw 95
    max_bw 95
    ltc 0.28765594293354313
  ]
  edge [
    source 37
    target 54
    bw 73
    max_bw 73
    ltc 0.339553367513547
  ]
  edge [
    source 37
    target 70
    bw 74
    max_bw 74
    ltc 0.057266681975993015
  ]
  edge [
    source 37
    target 99
    bw 54
    max_bw 54
    ltc 0.1905893439603281
  ]
  edge [
    source 38
    target 40
    bw 86
    max_bw 86
    ltc 0.1480017146940236
  ]
  edge [
    source 38
    target 41
    bw 94
    max_bw 94
    ltc 0.5317669138565126
  ]
  edge [
    source 38
    target 43
    bw 99
    max_bw 99
    ltc 0.060523684221801766
  ]
  edge [
    source 38
    target 45
    bw 73
    max_bw 73
    ltc 0.3538072932491811
  ]
  edge [
    source 38
    target 47
    bw 80
    max_bw 80
    ltc 0.24201113454331066
  ]
  edge [
    source 38
    target 65
    bw 75
    max_bw 75
    ltc 0.07406202732054853
  ]
  edge [
    source 38
    target 75
    bw 70
    max_bw 70
    ltc 0.12360807868878784
  ]
  edge [
    source 38
    target 91
    bw 83
    max_bw 83
    ltc 0.49055308757463895
  ]
  edge [
    source 39
    target 43
    bw 87
    max_bw 87
    ltc 0.1119729398966852
  ]
  edge [
    source 39
    target 46
    bw 73
    max_bw 73
    ltc 0.5067139478362224
  ]
  edge [
    source 39
    target 91
    bw 77
    max_bw 77
    ltc 0.4996832385594475
  ]
  edge [
    source 40
    target 47
    bw 79
    max_bw 79
    ltc 0.09827678520578881
  ]
  edge [
    source 40
    target 60
    bw 83
    max_bw 83
    ltc 0.6218639406404671
  ]
  edge [
    source 40
    target 85
    bw 72
    max_bw 72
    ltc 0.6076898415481689
  ]
  edge [
    source 41
    target 44
    bw 57
    max_bw 57
    ltc 0.40624047407691505
  ]
  edge [
    source 41
    target 46
    bw 91
    max_bw 91
    ltc 0.2836201154587941
  ]
  edge [
    source 41
    target 57
    bw 99
    max_bw 99
    ltc 0.39609242045568854
  ]
  edge [
    source 41
    target 66
    bw 59
    max_bw 59
    ltc 0.5189536208184463
  ]
  edge [
    source 41
    target 67
    bw 50
    max_bw 50
    ltc 0.4583320269929867
  ]
  edge [
    source 41
    target 68
    bw 69
    max_bw 69
    ltc 0.40938954420481993
  ]
  edge [
    source 41
    target 94
    bw 86
    max_bw 86
    ltc 0.2560962193701241
  ]
  edge [
    source 41
    target 95
    bw 67
    max_bw 67
    ltc 0.09762228221469758
  ]
  edge [
    source 42
    target 45
    bw 70
    max_bw 70
    ltc 0.39015921789445934
  ]
  edge [
    source 42
    target 70
    bw 98
    max_bw 98
    ltc 0.31060922534843444
  ]
  edge [
    source 42
    target 77
    bw 53
    max_bw 53
    ltc 0.1551952508148144
  ]
  edge [
    source 42
    target 80
    bw 92
    max_bw 92
    ltc 0.269489942215976
  ]
  edge [
    source 42
    target 98
    bw 51
    max_bw 51
    ltc 0.14263425334111784
  ]
  edge [
    source 43
    target 50
    bw 70
    max_bw 70
    ltc 0.18153490260624197
  ]
  edge [
    source 43
    target 53
    bw 86
    max_bw 86
    ltc 0.5172307699383129
  ]
  edge [
    source 43
    target 57
    bw 54
    max_bw 54
    ltc 0.33028258056539656
  ]
  edge [
    source 43
    target 61
    bw 66
    max_bw 66
    ltc 0.4605229342334579
  ]
  edge [
    source 43
    target 73
    bw 98
    max_bw 98
    ltc 0.08080468734983334
  ]
  edge [
    source 43
    target 89
    bw 97
    max_bw 97
    ltc 0.46279723865024996
  ]
  edge [
    source 43
    target 96
    bw 60
    max_bw 60
    ltc 0.2884768403326811
  ]
  edge [
    source 43
    target 98
    bw 95
    max_bw 95
    ltc 0.445017437097027
  ]
  edge [
    source 44
    target 45
    bw 90
    max_bw 90
    ltc 0.05565123084530967
  ]
  edge [
    source 44
    target 49
    bw 80
    max_bw 80
    ltc 0.300584493094454
  ]
  edge [
    source 44
    target 75
    bw 77
    max_bw 77
    ltc 0.43749273616158385
  ]
  edge [
    source 44
    target 83
    bw 93
    max_bw 93
    ltc 0.34369184469262454
  ]
  edge [
    source 44
    target 90
    bw 81
    max_bw 81
    ltc 0.2631562537916838
  ]
  edge [
    source 44
    target 91
    bw 70
    max_bw 70
    ltc 0.2700151121105422
  ]
  edge [
    source 45
    target 49
    bw 56
    max_bw 56
    ltc 0.34964614316601306
  ]
  edge [
    source 45
    target 57
    bw 59
    max_bw 59
    ltc 0.06041483782610054
  ]
  edge [
    source 45
    target 69
    bw 77
    max_bw 77
    ltc 0.7203245395715767
  ]
  edge [
    source 45
    target 72
    bw 97
    max_bw 97
    ltc 0.7823148743216192
  ]
  edge [
    source 45
    target 75
    bw 85
    max_bw 85
    ltc 0.47587163919464726
  ]
  edge [
    source 45
    target 87
    bw 69
    max_bw 69
    ltc 0.8018941779491828
  ]
  edge [
    source 45
    target 98
    bw 62
    max_bw 62
    ltc 0.29587826989655486
  ]
  edge [
    source 46
    target 60
    bw 68
    max_bw 68
    ltc 0.2968214363946256
  ]
  edge [
    source 46
    target 61
    bw 99
    max_bw 99
    ltc 0.3302144354226559
  ]
  edge [
    source 46
    target 65
    bw 60
    max_bw 60
    ltc 0.44632333873792107
  ]
  edge [
    source 46
    target 72
    bw 84
    max_bw 84
    ltc 0.3925837725150581
  ]
  edge [
    source 46
    target 78
    bw 83
    max_bw 83
    ltc 0.20970629605227417
  ]
  edge [
    source 47
    target 60
    bw 93
    max_bw 93
    ltc 0.6730899912113898
  ]
  edge [
    source 48
    target 60
    bw 53
    max_bw 53
    ltc 0.37766464034395836
  ]
  edge [
    source 48
    target 69
    bw 62
    max_bw 62
    ltc 0.4537287476478859
  ]
  edge [
    source 48
    target 70
    bw 86
    max_bw 86
    ltc 0.29286170962626384
  ]
  edge [
    source 48
    target 74
    bw 51
    max_bw 51
    ltc 0.15245410223236816
  ]
  edge [
    source 48
    target 90
    bw 99
    max_bw 99
    ltc 0.5618422722937081
  ]
  edge [
    source 48
    target 95
    bw 50
    max_bw 50
    ltc 0.33480104974198865
  ]
  edge [
    source 48
    target 99
    bw 89
    max_bw 89
    ltc 0.13587401880677652
  ]
  edge [
    source 49
    target 55
    bw 74
    max_bw 74
    ltc 0.5120239552350221
  ]
  edge [
    source 49
    target 60
    bw 86
    max_bw 86
    ltc 0.10210840413850167
  ]
  edge [
    source 49
    target 76
    bw 85
    max_bw 85
    ltc 0.5862897839270262
  ]
  edge [
    source 49
    target 80
    bw 55
    max_bw 55
    ltc 0.16057298123299793
  ]
  edge [
    source 49
    target 84
    bw 56
    max_bw 56
    ltc 0.47687160419713504
  ]
  edge [
    source 49
    target 92
    bw 53
    max_bw 53
    ltc 0.25762831983673257
  ]
  edge [
    source 50
    target 55
    bw 84
    max_bw 84
    ltc 0.505747424573345
  ]
  edge [
    source 50
    target 73
    bw 90
    max_bw 90
    ltc 0.1016753392275926
  ]
  edge [
    source 50
    target 75
    bw 83
    max_bw 83
    ltc 0.12868938335424082
  ]
  edge [
    source 50
    target 80
    bw 78
    max_bw 78
    ltc 0.4631457946087128
  ]
  edge [
    source 50
    target 93
    bw 54
    max_bw 54
    ltc 0.9062412106889544
  ]
  edge [
    source 51
    target 68
    bw 76
    max_bw 76
    ltc 0.08743058793859854
  ]
  edge [
    source 51
    target 82
    bw 82
    max_bw 82
    ltc 0.3805372155946946
  ]
  edge [
    source 51
    target 90
    bw 95
    max_bw 95
    ltc 0.40601347195526816
  ]
  edge [
    source 52
    target 62
    bw 59
    max_bw 59
    ltc 0.2145933492829749
  ]
  edge [
    source 52
    target 64
    bw 55
    max_bw 55
    ltc 0.011975114569928171
  ]
  edge [
    source 52
    target 72
    bw 83
    max_bw 83
    ltc 0.16071881836990393
  ]
  edge [
    source 52
    target 79
    bw 57
    max_bw 57
    ltc 0.3283671257772069
  ]
  edge [
    source 52
    target 86
    bw 80
    max_bw 80
    ltc 0.0935641657546255
  ]
  edge [
    source 52
    target 93
    bw 58
    max_bw 58
    ltc 0.11898809668425889
  ]
  edge [
    source 52
    target 94
    bw 70
    max_bw 70
    ltc 0.21809270980387396
  ]
  edge [
    source 53
    target 81
    bw 57
    max_bw 57
    ltc 0.2860687870992194
  ]
  edge [
    source 53
    target 84
    bw 53
    max_bw 53
    ltc 0.39649947042901507
  ]
  edge [
    source 53
    target 95
    bw 71
    max_bw 71
    ltc 0.1638077657760115
  ]
  edge [
    source 53
    target 98
    bw 77
    max_bw 77
    ltc 0.12627758658599791
  ]
  edge [
    source 54
    target 60
    bw 94
    max_bw 94
    ltc 0.3768691163010834
  ]
  edge [
    source 54
    target 73
    bw 53
    max_bw 53
    ltc 0.7425216314606475
  ]
  edge [
    source 54
    target 91
    bw 88
    max_bw 88
    ltc 0.3044062743523675
  ]
  edge [
    source 55
    target 56
    bw 70
    max_bw 70
    ltc 0.1598553357119269
  ]
  edge [
    source 55
    target 98
    bw 57
    max_bw 57
    ltc 0.6218015596189991
  ]
  edge [
    source 56
    target 59
    bw 69
    max_bw 69
    ltc 0.5906591267131873
  ]
  edge [
    source 56
    target 62
    bw 81
    max_bw 81
    ltc 0.5523003095390424
  ]
  edge [
    source 56
    target 72
    bw 50
    max_bw 50
    ltc 0.26478711862461984
  ]
  edge [
    source 56
    target 80
    bw 55
    max_bw 55
    ltc 0.4568289316948741
  ]
  edge [
    source 57
    target 77
    bw 77
    max_bw 77
    ltc 0.2024649817408895
  ]
  edge [
    source 57
    target 96
    bw 93
    max_bw 93
    ltc 0.048471502059176504
  ]
  edge [
    source 58
    target 88
    bw 80
    max_bw 80
    ltc 0.4195690793379177
  ]
  edge [
    source 58
    target 92
    bw 59
    max_bw 59
    ltc 0.6785225927930519
  ]
  edge [
    source 58
    target 99
    bw 69
    max_bw 69
    ltc 0.027232956733599403
  ]
  edge [
    source 59
    target 60
    bw 57
    max_bw 57
    ltc 0.22308526167347403
  ]
  edge [
    source 59
    target 68
    bw 71
    max_bw 71
    ltc 0.41615459502993546
  ]
  edge [
    source 59
    target 77
    bw 87
    max_bw 87
    ltc 0.3376214020208518
  ]
  edge [
    source 59
    target 79
    bw 78
    max_bw 78
    ltc 0.4725848460229121
  ]
  edge [
    source 59
    target 81
    bw 58
    max_bw 58
    ltc 0.1178723551241436
  ]
  edge [
    source 59
    target 83
    bw 93
    max_bw 93
    ltc 0.40912692080666263
  ]
  edge [
    source 59
    target 90
    bw 96
    max_bw 96
    ltc 0.014791514604834605
  ]
  edge [
    source 60
    target 75
    bw 50
    max_bw 50
    ltc 0.58340725743986
  ]
  edge [
    source 60
    target 80
    bw 90
    max_bw 90
    ltc 0.10598949963739132
  ]
  edge [
    source 60
    target 81
    bw 88
    max_bw 88
    ltc 0.11758983701083177
  ]
  edge [
    source 60
    target 89
    bw 75
    max_bw 75
    ltc 0.5964728323291855
  ]
  edge [
    source 60
    target 98
    bw 60
    max_bw 60
    ltc 0.12874340217831462
  ]
  edge [
    source 61
    target 78
    bw 84
    max_bw 84
    ltc 0.40188804024733255
  ]
  edge [
    source 61
    target 89
    bw 73
    max_bw 73
    ltc 0.035346046235073376
  ]
  edge [
    source 62
    target 64
    bw 82
    max_bw 82
    ltc 0.20821892520612037
  ]
  edge [
    source 62
    target 86
    bw 69
    max_bw 69
    ltc 0.25941357076920885
  ]
  edge [
    source 62
    target 89
    bw 76
    max_bw 76
    ltc 0.9799638150567778
  ]
  edge [
    source 63
    target 77
    bw 64
    max_bw 64
    ltc 0.19524592257027218
  ]
  edge [
    source 63
    target 83
    bw 82
    max_bw 82
    ltc 0.1398963904367821
  ]
  edge [
    source 63
    target 97
    bw 56
    max_bw 56
    ltc 0.5882311947369676
  ]
  edge [
    source 63
    target 98
    bw 83
    max_bw 83
    ltc 0.34870218324817015
  ]
  edge [
    source 64
    target 69
    bw 94
    max_bw 94
    ltc 0.22841191161342664
  ]
  edge [
    source 64
    target 72
    bw 100
    max_bw 100
    ltc 0.16610848501607706
  ]
  edge [
    source 64
    target 86
    bw 95
    max_bw 95
    ltc 0.10515730011256842
  ]
  edge [
    source 64
    target 93
    bw 91
    max_bw 91
    ltc 0.12372683511783888
  ]
  edge [
    source 65
    target 73
    bw 54
    max_bw 54
    ltc 0.1204651254128421
  ]
  edge [
    source 65
    target 98
    bw 79
    max_bw 79
    ltc 0.5280894402024483
  ]
  edge [
    source 66
    target 71
    bw 77
    max_bw 77
    ltc 0.5680165234647903
  ]
  edge [
    source 66
    target 78
    bw 67
    max_bw 67
    ltc 0.2850274146295944
  ]
  edge [
    source 66
    target 82
    bw 85
    max_bw 85
    ltc 0.29667208103880255
  ]
  edge [
    source 66
    target 92
    bw 52
    max_bw 52
    ltc 0.4517738990374556
  ]
  edge [
    source 67
    target 72
    bw 70
    max_bw 70
    ltc 0.1389412005108397
  ]
  edge [
    source 67
    target 76
    bw 95
    max_bw 95
    ltc 0.4185432537260358
  ]
  edge [
    source 67
    target 97
    bw 65
    max_bw 65
    ltc 0.3437629953452428
  ]
  edge [
    source 68
    target 69
    bw 86
    max_bw 86
    ltc 0.4842354423574204
  ]
  edge [
    source 68
    target 71
    bw 91
    max_bw 91
    ltc 0.3617596742815783
  ]
  edge [
    source 68
    target 78
    bw 54
    max_bw 54
    ltc 0.058351991422683026
  ]
  edge [
    source 68
    target 79
    bw 99
    max_bw 99
    ltc 0.5214784726679417
  ]
  edge [
    source 68
    target 82
    bw 100
    max_bw 100
    ltc 0.3036122269604838
  ]
  edge [
    source 68
    target 86
    bw 63
    max_bw 63
    ltc 0.40578152048152877
  ]
  edge [
    source 68
    target 88
    bw 80
    max_bw 80
    ltc 0.382335338856209
  ]
  edge [
    source 68
    target 97
    bw 95
    max_bw 95
    ltc 0.3119774926174429
  ]
  edge [
    source 69
    target 72
    bw 73
    max_bw 73
    ltc 0.23356134940483972
  ]
  edge [
    source 69
    target 79
    bw 84
    max_bw 84
    ltc 0.09218419166370902
  ]
  edge [
    source 69
    target 83
    bw 85
    max_bw 85
    ltc 0.4924563323046749
  ]
  edge [
    source 69
    target 84
    bw 59
    max_bw 59
    ltc 0.2690965351168122
  ]
  edge [
    source 69
    target 91
    bw 76
    max_bw 76
    ltc 0.4202399893494855
  ]
  edge [
    source 70
    target 95
    bw 76
    max_bw 76
    ltc 0.36468645869697236
  ]
  edge [
    source 71
    target 78
    bw 72
    max_bw 72
    ltc 0.4191158523196005
  ]
  edge [
    source 72
    target 82
    bw 62
    max_bw 62
    ltc 0.11209899840265926
  ]
  edge [
    source 72
    target 89
    bw 65
    max_bw 65
    ltc 0.6096777061172275
  ]
  edge [
    source 72
    target 92
    bw 84
    max_bw 84
    ltc 0.3628745259216287
  ]
  edge [
    source 73
    target 78
    bw 76
    max_bw 76
    ltc 0.5381585243245246
  ]
  edge [
    source 73
    target 79
    bw 88
    max_bw 88
    ltc 0.7238280434784203
  ]
  edge [
    source 74
    target 90
    bw 96
    max_bw 96
    ltc 0.5339132098611525
  ]
  edge [
    source 74
    target 96
    bw 66
    max_bw 66
    ltc 0.4010450372927977
  ]
  edge [
    source 76
    target 78
    bw 97
    max_bw 97
    ltc 0.20836216313682118
  ]
  edge [
    source 76
    target 87
    bw 90
    max_bw 90
    ltc 0.3314239011016271
  ]
  edge [
    source 76
    target 91
    bw 50
    max_bw 50
    ltc 0.7179460588851939
  ]
  edge [
    source 77
    target 85
    bw 60
    max_bw 60
    ltc 0.10404698971296863
  ]
  edge [
    source 77
    target 91
    bw 61
    max_bw 61
    ltc 0.1376165698206109
  ]
  edge [
    source 77
    target 97
    bw 63
    max_bw 63
    ltc 0.4351680506570264
  ]
  edge [
    source 78
    target 92
    bw 57
    max_bw 57
    ltc 0.1724186932115879
  ]
  edge [
    source 78
    target 98
    bw 80
    max_bw 80
    ltc 0.48519559266891904
  ]
  edge [
    source 79
    target 82
    bw 56
    max_bw 56
    ltc 0.22443589833314906
  ]
  edge [
    source 79
    target 86
    bw 95
    max_bw 95
    ltc 0.4105544754414383
  ]
  edge [
    source 80
    target 81
    bw 80
    max_bw 80
    ltc 0.13872429139634349
  ]
  edge [
    source 80
    target 91
    bw 65
    max_bw 65
    ltc 0.24497373732748262
  ]
  edge [
    source 80
    target 94
    bw 91
    max_bw 91
    ltc 0.21885382828477834
  ]
  edge [
    source 80
    target 97
    bw 88
    max_bw 88
    ltc 0.05772659706543374
  ]
  edge [
    source 82
    target 97
    bw 93
    max_bw 93
    ltc 0.1124110391479379
  ]
  edge [
    source 83
    target 85
    bw 68
    max_bw 68
    ltc 0.1282767868711127
  ]
  edge [
    source 83
    target 88
    bw 57
    max_bw 57
    ltc 0.48683948661393495
  ]
  edge [
    source 84
    target 97
    bw 78
    max_bw 78
    ltc 0.4163419894630959
  ]
  edge [
    source 84
    target 98
    bw 51
    max_bw 51
    ltc 0.4296643564981311
  ]
  edge [
    source 84
    target 99
    bw 91
    max_bw 91
    ltc 0.10724340097574597
  ]
  edge [
    source 85
    target 91
    bw 52
    max_bw 52
    ltc 0.05820776903585742
  ]
  edge [
    source 87
    target 95
    bw 100
    max_bw 100
    ltc 0.6984635867757697
  ]
  edge [
    source 88
    target 95
    bw 78
    max_bw 78
    ltc 0.2992713483170768
  ]
  edge [
    source 89
    target 91
    bw 64
    max_bw 64
    ltc 0.6628652709397022
  ]
  edge [
    source 90
    target 98
    bw 87
    max_bw 87
    ltc 0.21397048565752047
  ]
  edge [
    source 91
    target 98
    bw 77
    max_bw 77
    ltc 0.03320782206216827
  ]
  edge [
    source 92
    target 93
    bw 86
    max_bw 86
    ltc 0.6170290902922798
  ]
  edge [
    source 93
    target 99
    bw 81
    max_bw 81
    ltc 0.6765101555123878
  ]
  edge [
    source 96
    target 99
    bw 79
    max_bw 79
    ltc 0.5632889613091199
  ]
]
