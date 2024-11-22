; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @printFloat(double)

declare void @printString(i64, ptr)

declare double @system_time_msec()

declare double @generate_random()

define i32 @main() {
  call void @printFloat(double 1.000000e+00)
  br label %1

1:                                                ; preds = %10, %0
  %2 = phi i64 [ %11, %10 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 64
  br i1 %3, label %4, label %12

4:                                                ; preds = %7, %1
  %5 = phi i64 [ %9, %7 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 64
  br i1 %6, label %7, label %10

7:                                                ; preds = %4
  %8 = call double @generate_random()
  %9 = add i64 %5, 1
  br label %4

10:                                               ; preds = %4
  %11 = add i64 %2, 1
  br label %1

12:                                               ; preds = %21, %1
  %13 = phi i64 [ %22, %21 ], [ 0, %1 ]
  %14 = icmp slt i64 %13, 64
  br i1 %14, label %15, label %23

15:                                               ; preds = %18, %12
  %16 = phi i64 [ %20, %18 ], [ 0, %12 ]
  %17 = icmp slt i64 %16, 64
  br i1 %17, label %18, label %21

18:                                               ; preds = %15
  %19 = call double @generate_random()
  %20 = add i64 %16, 1
  br label %15

21:                                               ; preds = %15
  %22 = add i64 %13, 1
  br label %12

23:                                               ; preds = %36, %12
  %24 = phi i64 [ %37, %36 ], [ 0, %12 ]
  %25 = icmp slt i64 %24, 64
  br i1 %25, label %26, label %38

26:                                               ; preds = %34, %23
  %27 = phi i64 [ %35, %34 ], [ 0, %23 ]
  %28 = icmp slt i64 %27, 64
  br i1 %28, label %29, label %36

29:                                               ; preds = %32, %26
  %30 = phi i64 [ %33, %32 ], [ 0, %26 ]
  %31 = icmp slt i64 %30, 64
  br i1 %31, label %32, label %34

32:                                               ; preds = %29
  %33 = add i64 %30, 1
  br label %29

34:                                               ; preds = %29
  %35 = add i64 %27, 1
  br label %26

36:                                               ; preds = %26
  %37 = add i64 %24, 1
  br label %23

38:                                               ; preds = %23
  call void @printFloat(double 2.000000e+00)
  call void @printFloat(double 3.000000e+00)
  br label %39

39:                                               ; preds = %48, %38
  %40 = phi i64 [ %49, %48 ], [ 0, %38 ]
  %41 = icmp slt i64 %40, 128
  br i1 %41, label %42, label %50

42:                                               ; preds = %45, %39
  %43 = phi i64 [ %47, %45 ], [ 0, %39 ]
  %44 = icmp slt i64 %43, 128
  br i1 %44, label %45, label %48

45:                                               ; preds = %42
  %46 = call double @generate_random()
  %47 = add i64 %43, 1
  br label %42

48:                                               ; preds = %42
  %49 = add i64 %40, 1
  br label %39

50:                                               ; preds = %59, %39
  %51 = phi i64 [ %60, %59 ], [ 0, %39 ]
  %52 = icmp slt i64 %51, 128
  br i1 %52, label %53, label %61

53:                                               ; preds = %56, %50
  %54 = phi i64 [ %58, %56 ], [ 0, %50 ]
  %55 = icmp slt i64 %54, 128
  br i1 %55, label %56, label %59

56:                                               ; preds = %53
  %57 = call double @generate_random()
  %58 = add i64 %54, 1
  br label %53

59:                                               ; preds = %53
  %60 = add i64 %51, 1
  br label %50

61:                                               ; preds = %74, %50
  %62 = phi i64 [ %75, %74 ], [ 0, %50 ]
  %63 = icmp slt i64 %62, 128
  br i1 %63, label %64, label %76

64:                                               ; preds = %72, %61
  %65 = phi i64 [ %73, %72 ], [ 0, %61 ]
  %66 = icmp slt i64 %65, 128
  br i1 %66, label %67, label %74

67:                                               ; preds = %70, %64
  %68 = phi i64 [ %71, %70 ], [ 0, %64 ]
  %69 = icmp slt i64 %68, 128
  br i1 %69, label %70, label %72

70:                                               ; preds = %67
  %71 = add i64 %68, 1
  br label %67

72:                                               ; preds = %67
  %73 = add i64 %65, 1
  br label %64

74:                                               ; preds = %64
  %75 = add i64 %62, 1
  br label %61

76:                                               ; preds = %61
  call void @printFloat(double 4.000000e+00)
  call void @printFloat(double 5.000000e+00)
  br label %77

77:                                               ; preds = %86, %76
  %78 = phi i64 [ %87, %86 ], [ 0, %76 ]
  %79 = icmp slt i64 %78, 256
  br i1 %79, label %80, label %88

80:                                               ; preds = %83, %77
  %81 = phi i64 [ %85, %83 ], [ 0, %77 ]
  %82 = icmp slt i64 %81, 256
  br i1 %82, label %83, label %86

83:                                               ; preds = %80
  %84 = call double @generate_random()
  %85 = add i64 %81, 1
  br label %80

86:                                               ; preds = %80
  %87 = add i64 %78, 1
  br label %77

88:                                               ; preds = %97, %77
  %89 = phi i64 [ %98, %97 ], [ 0, %77 ]
  %90 = icmp slt i64 %89, 256
  br i1 %90, label %91, label %99

91:                                               ; preds = %94, %88
  %92 = phi i64 [ %96, %94 ], [ 0, %88 ]
  %93 = icmp slt i64 %92, 256
  br i1 %93, label %94, label %97

94:                                               ; preds = %91
  %95 = call double @generate_random()
  %96 = add i64 %92, 1
  br label %91

97:                                               ; preds = %91
  %98 = add i64 %89, 1
  br label %88

99:                                               ; preds = %112, %88
  %100 = phi i64 [ %113, %112 ], [ 0, %88 ]
  %101 = icmp slt i64 %100, 256
  br i1 %101, label %102, label %114

102:                                              ; preds = %110, %99
  %103 = phi i64 [ %111, %110 ], [ 0, %99 ]
  %104 = icmp slt i64 %103, 256
  br i1 %104, label %105, label %112

105:                                              ; preds = %108, %102
  %106 = phi i64 [ %109, %108 ], [ 0, %102 ]
  %107 = icmp slt i64 %106, 256
  br i1 %107, label %108, label %110

108:                                              ; preds = %105
  %109 = add i64 %106, 1
  br label %105

110:                                              ; preds = %105
  %111 = add i64 %103, 1
  br label %102

112:                                              ; preds = %102
  %113 = add i64 %100, 1
  br label %99

114:                                              ; preds = %99
  call void @printFloat(double 6.000000e+00)
  call void @printFloat(double 7.000000e+00)
  br label %115

115:                                              ; preds = %124, %114
  %116 = phi i64 [ %125, %124 ], [ 0, %114 ]
  %117 = icmp slt i64 %116, 512
  br i1 %117, label %118, label %126

118:                                              ; preds = %121, %115
  %119 = phi i64 [ %123, %121 ], [ 0, %115 ]
  %120 = icmp slt i64 %119, 512
  br i1 %120, label %121, label %124

121:                                              ; preds = %118
  %122 = call double @generate_random()
  %123 = add i64 %119, 1
  br label %118

124:                                              ; preds = %118
  %125 = add i64 %116, 1
  br label %115

126:                                              ; preds = %135, %115
  %127 = phi i64 [ %136, %135 ], [ 0, %115 ]
  %128 = icmp slt i64 %127, 512
  br i1 %128, label %129, label %137

129:                                              ; preds = %132, %126
  %130 = phi i64 [ %134, %132 ], [ 0, %126 ]
  %131 = icmp slt i64 %130, 512
  br i1 %131, label %132, label %135

132:                                              ; preds = %129
  %133 = call double @generate_random()
  %134 = add i64 %130, 1
  br label %129

135:                                              ; preds = %129
  %136 = add i64 %127, 1
  br label %126

137:                                              ; preds = %150, %126
  %138 = phi i64 [ %151, %150 ], [ 0, %126 ]
  %139 = icmp slt i64 %138, 512
  br i1 %139, label %140, label %152

140:                                              ; preds = %148, %137
  %141 = phi i64 [ %149, %148 ], [ 0, %137 ]
  %142 = icmp slt i64 %141, 512
  br i1 %142, label %143, label %150

143:                                              ; preds = %146, %140
  %144 = phi i64 [ %147, %146 ], [ 0, %140 ]
  %145 = icmp slt i64 %144, 512
  br i1 %145, label %146, label %148

146:                                              ; preds = %143
  %147 = add i64 %144, 1
  br label %143

148:                                              ; preds = %143
  %149 = add i64 %141, 1
  br label %140

150:                                              ; preds = %140
  %151 = add i64 %138, 1
  br label %137

152:                                              ; preds = %137
  call void @printFloat(double 8.000000e+00)
  call void @printFloat(double 9.000000e+00)
  br label %153

153:                                              ; preds = %162, %152
  %154 = phi i64 [ %163, %162 ], [ 0, %152 ]
  %155 = icmp slt i64 %154, 1024
  br i1 %155, label %156, label %164

156:                                              ; preds = %159, %153
  %157 = phi i64 [ %161, %159 ], [ 0, %153 ]
  %158 = icmp slt i64 %157, 1024
  br i1 %158, label %159, label %162

159:                                              ; preds = %156
  %160 = call double @generate_random()
  %161 = add i64 %157, 1
  br label %156

162:                                              ; preds = %156
  %163 = add i64 %154, 1
  br label %153

164:                                              ; preds = %173, %153
  %165 = phi i64 [ %174, %173 ], [ 0, %153 ]
  %166 = icmp slt i64 %165, 1024
  br i1 %166, label %167, label %175

167:                                              ; preds = %170, %164
  %168 = phi i64 [ %172, %170 ], [ 0, %164 ]
  %169 = icmp slt i64 %168, 1024
  br i1 %169, label %170, label %173

170:                                              ; preds = %167
  %171 = call double @generate_random()
  %172 = add i64 %168, 1
  br label %167

173:                                              ; preds = %167
  %174 = add i64 %165, 1
  br label %164

175:                                              ; preds = %188, %164
  %176 = phi i64 [ %189, %188 ], [ 0, %164 ]
  %177 = icmp slt i64 %176, 1024
  br i1 %177, label %178, label %190

178:                                              ; preds = %186, %175
  %179 = phi i64 [ %187, %186 ], [ 0, %175 ]
  %180 = icmp slt i64 %179, 1024
  br i1 %180, label %181, label %188

181:                                              ; preds = %184, %178
  %182 = phi i64 [ %185, %184 ], [ 0, %178 ]
  %183 = icmp slt i64 %182, 1024
  br i1 %183, label %184, label %186

184:                                              ; preds = %181
  %185 = add i64 %182, 1
  br label %181

186:                                              ; preds = %181
  %187 = add i64 %179, 1
  br label %178

188:                                              ; preds = %178
  %189 = add i64 %176, 1
  br label %175

190:                                              ; preds = %175
  call void @printFloat(double 1.000000e+01)
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
