; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @printFloat(double)

declare void @printString(i64, ptr)

declare double @system_time_msec()

declare double @generate_random()

define i32 @main() {
  call void @printFloat(double 1.000000e+00)
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 4096) to i64))
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 64, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 64, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 64, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  br label %9

9:                                                ; preds = %22, %0
  %10 = phi i64 [ %23, %22 ], [ 0, %0 ]
  %11 = icmp slt i64 %10, 64
  br i1 %11, label %12, label %24

12:                                               ; preds = %15, %9
  %13 = phi i64 [ %21, %15 ], [ 0, %9 ]
  %14 = icmp slt i64 %13, 64
  br i1 %14, label %15, label %22

15:                                               ; preds = %12
  %16 = call double @generate_random()
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %18 = mul i64 %10, 64
  %19 = add i64 %18, %13
  %20 = getelementptr double, ptr %17, i64 %19
  store double %16, ptr %20, align 8
  %21 = add i64 %13, 1
  br label %12

22:                                               ; preds = %12
  %23 = add i64 %10, 1
  br label %9

24:                                               ; preds = %9
  %25 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 4096) to i64))
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %25, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, ptr %25, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 0, 2
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 64, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 64, 3, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 64, 4, 0
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 1, 4, 1
  br label %33

33:                                               ; preds = %46, %24
  %34 = phi i64 [ %47, %46 ], [ 0, %24 ]
  %35 = icmp slt i64 %34, 64
  br i1 %35, label %36, label %48

36:                                               ; preds = %39, %33
  %37 = phi i64 [ %45, %39 ], [ 0, %33 ]
  %38 = icmp slt i64 %37, 64
  br i1 %38, label %39, label %46

39:                                               ; preds = %36
  %40 = call double @generate_random()
  %41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %42 = mul i64 %34, 64
  %43 = add i64 %42, %37
  %44 = getelementptr double, ptr %41, i64 %43
  store double %40, ptr %44, align 8
  %45 = add i64 %37, 1
  br label %36

46:                                               ; preds = %36
  %47 = add i64 %34, 1
  br label %33

48:                                               ; preds = %33
  %49 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 4096) to i64))
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %49, 0
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, ptr %49, 1
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 0, 2
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 64, 3, 0
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 64, 3, 1
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 64, 4, 0
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 1, 4, 1
  br label %57

57:                                               ; preds = %69, %48
  %58 = phi i64 [ %70, %69 ], [ 0, %48 ]
  %59 = icmp slt i64 %58, 64
  br i1 %59, label %60, label %71

60:                                               ; preds = %63, %57
  %61 = phi i64 [ %68, %63 ], [ 0, %57 ]
  %62 = icmp slt i64 %61, 64
  br i1 %62, label %63, label %69

63:                                               ; preds = %60
  %64 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, 1
  %65 = mul i64 %58, 64
  %66 = add i64 %65, %61
  %67 = getelementptr double, ptr %64, i64 %66
  store double 0.000000e+00, ptr %67, align 8
  %68 = add i64 %61, 1
  br label %60

69:                                               ; preds = %60
  %70 = add i64 %58, 1
  br label %57

71:                                               ; preds = %105, %57
  %72 = phi i64 [ %106, %105 ], [ 0, %57 ]
  %73 = icmp slt i64 %72, 64
  br i1 %73, label %74, label %107

74:                                               ; preds = %103, %71
  %75 = phi i64 [ %104, %103 ], [ 0, %71 ]
  %76 = icmp slt i64 %75, 64
  br i1 %76, label %77, label %105

77:                                               ; preds = %80, %74
  %78 = phi i64 [ %102, %80 ], [ 0, %74 ]
  %79 = icmp slt i64 %78, 64
  br i1 %79, label %80, label %103

80:                                               ; preds = %77
  %81 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, 1
  %82 = mul i64 %72, 64
  %83 = add i64 %82, %75
  %84 = getelementptr double, ptr %81, i64 %83
  %85 = load double, ptr %84, align 8
  %86 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %87 = mul i64 %72, 64
  %88 = add i64 %87, %78
  %89 = getelementptr double, ptr %86, i64 %88
  %90 = load double, ptr %89, align 8
  %91 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %92 = mul i64 %78, 64
  %93 = add i64 %92, %75
  %94 = getelementptr double, ptr %91, i64 %93
  %95 = load double, ptr %94, align 8
  %96 = fmul double %90, %95
  %97 = fadd double %85, %96
  %98 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, 1
  %99 = mul i64 %72, 64
  %100 = add i64 %99, %75
  %101 = getelementptr double, ptr %98, i64 %100
  store double %97, ptr %101, align 8
  %102 = add i64 %78, 1
  br label %77

103:                                              ; preds = %77
  %104 = add i64 %75, 1
  br label %74

105:                                              ; preds = %74
  %106 = add i64 %72, 1
  br label %71

107:                                              ; preds = %71
  call void @printFloat(double 2.000000e+00)
  call void @printFloat(double 3.000000e+00)
  %108 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 16384) to i64))
  %109 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %108, 0
  %110 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, ptr %108, 1
  %111 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %110, i64 0, 2
  %112 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, i64 128, 3, 0
  %113 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, i64 128, 3, 1
  %114 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %113, i64 128, 4, 0
  %115 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, i64 1, 4, 1
  br label %116

116:                                              ; preds = %129, %107
  %117 = phi i64 [ %130, %129 ], [ 0, %107 ]
  %118 = icmp slt i64 %117, 128
  br i1 %118, label %119, label %131

119:                                              ; preds = %122, %116
  %120 = phi i64 [ %128, %122 ], [ 0, %116 ]
  %121 = icmp slt i64 %120, 128
  br i1 %121, label %122, label %129

122:                                              ; preds = %119
  %123 = call double @generate_random()
  %124 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, 1
  %125 = mul i64 %117, 128
  %126 = add i64 %125, %120
  %127 = getelementptr double, ptr %124, i64 %126
  store double %123, ptr %127, align 8
  %128 = add i64 %120, 1
  br label %119

129:                                              ; preds = %119
  %130 = add i64 %117, 1
  br label %116

131:                                              ; preds = %116
  %132 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 16384) to i64))
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %132, 0
  %134 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, ptr %132, 1
  %135 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %134, i64 0, 2
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %135, i64 128, 3, 0
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, i64 128, 3, 1
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %137, i64 128, 4, 0
  %139 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, i64 1, 4, 1
  br label %140

140:                                              ; preds = %153, %131
  %141 = phi i64 [ %154, %153 ], [ 0, %131 ]
  %142 = icmp slt i64 %141, 128
  br i1 %142, label %143, label %155

143:                                              ; preds = %146, %140
  %144 = phi i64 [ %152, %146 ], [ 0, %140 ]
  %145 = icmp slt i64 %144, 128
  br i1 %145, label %146, label %153

146:                                              ; preds = %143
  %147 = call double @generate_random()
  %148 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %139, 1
  %149 = mul i64 %141, 128
  %150 = add i64 %149, %144
  %151 = getelementptr double, ptr %148, i64 %150
  store double %147, ptr %151, align 8
  %152 = add i64 %144, 1
  br label %143

153:                                              ; preds = %143
  %154 = add i64 %141, 1
  br label %140

155:                                              ; preds = %140
  %156 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 16384) to i64))
  %157 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %156, 0
  %158 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %157, ptr %156, 1
  %159 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %158, i64 0, 2
  %160 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %159, i64 128, 3, 0
  %161 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %160, i64 128, 3, 1
  %162 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %161, i64 128, 4, 0
  %163 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %162, i64 1, 4, 1
  br label %164

164:                                              ; preds = %176, %155
  %165 = phi i64 [ %177, %176 ], [ 0, %155 ]
  %166 = icmp slt i64 %165, 128
  br i1 %166, label %167, label %178

167:                                              ; preds = %170, %164
  %168 = phi i64 [ %175, %170 ], [ 0, %164 ]
  %169 = icmp slt i64 %168, 128
  br i1 %169, label %170, label %176

170:                                              ; preds = %167
  %171 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %163, 1
  %172 = mul i64 %165, 128
  %173 = add i64 %172, %168
  %174 = getelementptr double, ptr %171, i64 %173
  store double 0.000000e+00, ptr %174, align 8
  %175 = add i64 %168, 1
  br label %167

176:                                              ; preds = %167
  %177 = add i64 %165, 1
  br label %164

178:                                              ; preds = %212, %164
  %179 = phi i64 [ %213, %212 ], [ 0, %164 ]
  %180 = icmp slt i64 %179, 128
  br i1 %180, label %181, label %214

181:                                              ; preds = %210, %178
  %182 = phi i64 [ %211, %210 ], [ 0, %178 ]
  %183 = icmp slt i64 %182, 128
  br i1 %183, label %184, label %212

184:                                              ; preds = %187, %181
  %185 = phi i64 [ %209, %187 ], [ 0, %181 ]
  %186 = icmp slt i64 %185, 128
  br i1 %186, label %187, label %210

187:                                              ; preds = %184
  %188 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %163, 1
  %189 = mul i64 %179, 128
  %190 = add i64 %189, %182
  %191 = getelementptr double, ptr %188, i64 %190
  %192 = load double, ptr %191, align 8
  %193 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, 1
  %194 = mul i64 %179, 128
  %195 = add i64 %194, %185
  %196 = getelementptr double, ptr %193, i64 %195
  %197 = load double, ptr %196, align 8
  %198 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %139, 1
  %199 = mul i64 %185, 128
  %200 = add i64 %199, %182
  %201 = getelementptr double, ptr %198, i64 %200
  %202 = load double, ptr %201, align 8
  %203 = fmul double %197, %202
  %204 = fadd double %192, %203
  %205 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %163, 1
  %206 = mul i64 %179, 128
  %207 = add i64 %206, %182
  %208 = getelementptr double, ptr %205, i64 %207
  store double %204, ptr %208, align 8
  %209 = add i64 %185, 1
  br label %184

210:                                              ; preds = %184
  %211 = add i64 %182, 1
  br label %181

212:                                              ; preds = %181
  %213 = add i64 %179, 1
  br label %178

214:                                              ; preds = %178
  call void @printFloat(double 4.000000e+00)
  call void @printFloat(double 5.000000e+00)
  %215 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 65536) to i64))
  %216 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %215, 0
  %217 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %216, ptr %215, 1
  %218 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %217, i64 0, 2
  %219 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %218, i64 256, 3, 0
  %220 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %219, i64 256, 3, 1
  %221 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %220, i64 256, 4, 0
  %222 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %221, i64 1, 4, 1
  br label %223

223:                                              ; preds = %236, %214
  %224 = phi i64 [ %237, %236 ], [ 0, %214 ]
  %225 = icmp slt i64 %224, 256
  br i1 %225, label %226, label %238

226:                                              ; preds = %229, %223
  %227 = phi i64 [ %235, %229 ], [ 0, %223 ]
  %228 = icmp slt i64 %227, 256
  br i1 %228, label %229, label %236

229:                                              ; preds = %226
  %230 = call double @generate_random()
  %231 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %222, 1
  %232 = mul i64 %224, 256
  %233 = add i64 %232, %227
  %234 = getelementptr double, ptr %231, i64 %233
  store double %230, ptr %234, align 8
  %235 = add i64 %227, 1
  br label %226

236:                                              ; preds = %226
  %237 = add i64 %224, 1
  br label %223

238:                                              ; preds = %223
  %239 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 65536) to i64))
  %240 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %239, 0
  %241 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %240, ptr %239, 1
  %242 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %241, i64 0, 2
  %243 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %242, i64 256, 3, 0
  %244 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %243, i64 256, 3, 1
  %245 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %244, i64 256, 4, 0
  %246 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %245, i64 1, 4, 1
  br label %247

247:                                              ; preds = %260, %238
  %248 = phi i64 [ %261, %260 ], [ 0, %238 ]
  %249 = icmp slt i64 %248, 256
  br i1 %249, label %250, label %262

250:                                              ; preds = %253, %247
  %251 = phi i64 [ %259, %253 ], [ 0, %247 ]
  %252 = icmp slt i64 %251, 256
  br i1 %252, label %253, label %260

253:                                              ; preds = %250
  %254 = call double @generate_random()
  %255 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %246, 1
  %256 = mul i64 %248, 256
  %257 = add i64 %256, %251
  %258 = getelementptr double, ptr %255, i64 %257
  store double %254, ptr %258, align 8
  %259 = add i64 %251, 1
  br label %250

260:                                              ; preds = %250
  %261 = add i64 %248, 1
  br label %247

262:                                              ; preds = %247
  %263 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 65536) to i64))
  %264 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %263, 0
  %265 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %264, ptr %263, 1
  %266 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %265, i64 0, 2
  %267 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %266, i64 256, 3, 0
  %268 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %267, i64 256, 3, 1
  %269 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %268, i64 256, 4, 0
  %270 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %269, i64 1, 4, 1
  br label %271

271:                                              ; preds = %283, %262
  %272 = phi i64 [ %284, %283 ], [ 0, %262 ]
  %273 = icmp slt i64 %272, 256
  br i1 %273, label %274, label %285

274:                                              ; preds = %277, %271
  %275 = phi i64 [ %282, %277 ], [ 0, %271 ]
  %276 = icmp slt i64 %275, 256
  br i1 %276, label %277, label %283

277:                                              ; preds = %274
  %278 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %270, 1
  %279 = mul i64 %272, 256
  %280 = add i64 %279, %275
  %281 = getelementptr double, ptr %278, i64 %280
  store double 0.000000e+00, ptr %281, align 8
  %282 = add i64 %275, 1
  br label %274

283:                                              ; preds = %274
  %284 = add i64 %272, 1
  br label %271

285:                                              ; preds = %319, %271
  %286 = phi i64 [ %320, %319 ], [ 0, %271 ]
  %287 = icmp slt i64 %286, 256
  br i1 %287, label %288, label %321

288:                                              ; preds = %317, %285
  %289 = phi i64 [ %318, %317 ], [ 0, %285 ]
  %290 = icmp slt i64 %289, 256
  br i1 %290, label %291, label %319

291:                                              ; preds = %294, %288
  %292 = phi i64 [ %316, %294 ], [ 0, %288 ]
  %293 = icmp slt i64 %292, 256
  br i1 %293, label %294, label %317

294:                                              ; preds = %291
  %295 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %270, 1
  %296 = mul i64 %286, 256
  %297 = add i64 %296, %289
  %298 = getelementptr double, ptr %295, i64 %297
  %299 = load double, ptr %298, align 8
  %300 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %222, 1
  %301 = mul i64 %286, 256
  %302 = add i64 %301, %292
  %303 = getelementptr double, ptr %300, i64 %302
  %304 = load double, ptr %303, align 8
  %305 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %246, 1
  %306 = mul i64 %292, 256
  %307 = add i64 %306, %289
  %308 = getelementptr double, ptr %305, i64 %307
  %309 = load double, ptr %308, align 8
  %310 = fmul double %304, %309
  %311 = fadd double %299, %310
  %312 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %270, 1
  %313 = mul i64 %286, 256
  %314 = add i64 %313, %289
  %315 = getelementptr double, ptr %312, i64 %314
  store double %311, ptr %315, align 8
  %316 = add i64 %292, 1
  br label %291

317:                                              ; preds = %291
  %318 = add i64 %289, 1
  br label %288

319:                                              ; preds = %288
  %320 = add i64 %286, 1
  br label %285

321:                                              ; preds = %285
  call void @printFloat(double 6.000000e+00)
  call void @printFloat(double 7.000000e+00)
  %322 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 262144) to i64))
  %323 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %322, 0
  %324 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %323, ptr %322, 1
  %325 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %324, i64 0, 2
  %326 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %325, i64 512, 3, 0
  %327 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %326, i64 512, 3, 1
  %328 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %327, i64 512, 4, 0
  %329 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %328, i64 1, 4, 1
  br label %330

330:                                              ; preds = %343, %321
  %331 = phi i64 [ %344, %343 ], [ 0, %321 ]
  %332 = icmp slt i64 %331, 512
  br i1 %332, label %333, label %345

333:                                              ; preds = %336, %330
  %334 = phi i64 [ %342, %336 ], [ 0, %330 ]
  %335 = icmp slt i64 %334, 512
  br i1 %335, label %336, label %343

336:                                              ; preds = %333
  %337 = call double @generate_random()
  %338 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %329, 1
  %339 = mul i64 %331, 512
  %340 = add i64 %339, %334
  %341 = getelementptr double, ptr %338, i64 %340
  store double %337, ptr %341, align 8
  %342 = add i64 %334, 1
  br label %333

343:                                              ; preds = %333
  %344 = add i64 %331, 1
  br label %330

345:                                              ; preds = %330
  %346 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 262144) to i64))
  %347 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %346, 0
  %348 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %347, ptr %346, 1
  %349 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %348, i64 0, 2
  %350 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %349, i64 512, 3, 0
  %351 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %350, i64 512, 3, 1
  %352 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %351, i64 512, 4, 0
  %353 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %352, i64 1, 4, 1
  br label %354

354:                                              ; preds = %367, %345
  %355 = phi i64 [ %368, %367 ], [ 0, %345 ]
  %356 = icmp slt i64 %355, 512
  br i1 %356, label %357, label %369

357:                                              ; preds = %360, %354
  %358 = phi i64 [ %366, %360 ], [ 0, %354 ]
  %359 = icmp slt i64 %358, 512
  br i1 %359, label %360, label %367

360:                                              ; preds = %357
  %361 = call double @generate_random()
  %362 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %353, 1
  %363 = mul i64 %355, 512
  %364 = add i64 %363, %358
  %365 = getelementptr double, ptr %362, i64 %364
  store double %361, ptr %365, align 8
  %366 = add i64 %358, 1
  br label %357

367:                                              ; preds = %357
  %368 = add i64 %355, 1
  br label %354

369:                                              ; preds = %354
  %370 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 262144) to i64))
  %371 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %370, 0
  %372 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %371, ptr %370, 1
  %373 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %372, i64 0, 2
  %374 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %373, i64 512, 3, 0
  %375 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %374, i64 512, 3, 1
  %376 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %375, i64 512, 4, 0
  %377 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %376, i64 1, 4, 1
  br label %378

378:                                              ; preds = %390, %369
  %379 = phi i64 [ %391, %390 ], [ 0, %369 ]
  %380 = icmp slt i64 %379, 512
  br i1 %380, label %381, label %392

381:                                              ; preds = %384, %378
  %382 = phi i64 [ %389, %384 ], [ 0, %378 ]
  %383 = icmp slt i64 %382, 512
  br i1 %383, label %384, label %390

384:                                              ; preds = %381
  %385 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %377, 1
  %386 = mul i64 %379, 512
  %387 = add i64 %386, %382
  %388 = getelementptr double, ptr %385, i64 %387
  store double 0.000000e+00, ptr %388, align 8
  %389 = add i64 %382, 1
  br label %381

390:                                              ; preds = %381
  %391 = add i64 %379, 1
  br label %378

392:                                              ; preds = %426, %378
  %393 = phi i64 [ %427, %426 ], [ 0, %378 ]
  %394 = icmp slt i64 %393, 512
  br i1 %394, label %395, label %428

395:                                              ; preds = %424, %392
  %396 = phi i64 [ %425, %424 ], [ 0, %392 ]
  %397 = icmp slt i64 %396, 512
  br i1 %397, label %398, label %426

398:                                              ; preds = %401, %395
  %399 = phi i64 [ %423, %401 ], [ 0, %395 ]
  %400 = icmp slt i64 %399, 512
  br i1 %400, label %401, label %424

401:                                              ; preds = %398
  %402 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %377, 1
  %403 = mul i64 %393, 512
  %404 = add i64 %403, %396
  %405 = getelementptr double, ptr %402, i64 %404
  %406 = load double, ptr %405, align 8
  %407 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %329, 1
  %408 = mul i64 %393, 512
  %409 = add i64 %408, %399
  %410 = getelementptr double, ptr %407, i64 %409
  %411 = load double, ptr %410, align 8
  %412 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %353, 1
  %413 = mul i64 %399, 512
  %414 = add i64 %413, %396
  %415 = getelementptr double, ptr %412, i64 %414
  %416 = load double, ptr %415, align 8
  %417 = fmul double %411, %416
  %418 = fadd double %406, %417
  %419 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %377, 1
  %420 = mul i64 %393, 512
  %421 = add i64 %420, %396
  %422 = getelementptr double, ptr %419, i64 %421
  store double %418, ptr %422, align 8
  %423 = add i64 %399, 1
  br label %398

424:                                              ; preds = %398
  %425 = add i64 %396, 1
  br label %395

426:                                              ; preds = %395
  %427 = add i64 %393, 1
  br label %392

428:                                              ; preds = %392
  call void @printFloat(double 8.000000e+00)
  call void @printFloat(double 9.000000e+00)
  %429 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 1048576) to i64))
  %430 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %429, 0
  %431 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %430, ptr %429, 1
  %432 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %431, i64 0, 2
  %433 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %432, i64 1024, 3, 0
  %434 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %433, i64 1024, 3, 1
  %435 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %434, i64 1024, 4, 0
  %436 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %435, i64 1, 4, 1
  br label %437

437:                                              ; preds = %450, %428
  %438 = phi i64 [ %451, %450 ], [ 0, %428 ]
  %439 = icmp slt i64 %438, 1024
  br i1 %439, label %440, label %452

440:                                              ; preds = %443, %437
  %441 = phi i64 [ %449, %443 ], [ 0, %437 ]
  %442 = icmp slt i64 %441, 1024
  br i1 %442, label %443, label %450

443:                                              ; preds = %440
  %444 = call double @generate_random()
  %445 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %436, 1
  %446 = mul i64 %438, 1024
  %447 = add i64 %446, %441
  %448 = getelementptr double, ptr %445, i64 %447
  store double %444, ptr %448, align 8
  %449 = add i64 %441, 1
  br label %440

450:                                              ; preds = %440
  %451 = add i64 %438, 1
  br label %437

452:                                              ; preds = %437
  %453 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 1048576) to i64))
  %454 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %453, 0
  %455 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %454, ptr %453, 1
  %456 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %455, i64 0, 2
  %457 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %456, i64 1024, 3, 0
  %458 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %457, i64 1024, 3, 1
  %459 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %458, i64 1024, 4, 0
  %460 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %459, i64 1, 4, 1
  br label %461

461:                                              ; preds = %474, %452
  %462 = phi i64 [ %475, %474 ], [ 0, %452 ]
  %463 = icmp slt i64 %462, 1024
  br i1 %463, label %464, label %476

464:                                              ; preds = %467, %461
  %465 = phi i64 [ %473, %467 ], [ 0, %461 ]
  %466 = icmp slt i64 %465, 1024
  br i1 %466, label %467, label %474

467:                                              ; preds = %464
  %468 = call double @generate_random()
  %469 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %460, 1
  %470 = mul i64 %462, 1024
  %471 = add i64 %470, %465
  %472 = getelementptr double, ptr %469, i64 %471
  store double %468, ptr %472, align 8
  %473 = add i64 %465, 1
  br label %464

474:                                              ; preds = %464
  %475 = add i64 %462, 1
  br label %461

476:                                              ; preds = %461
  %477 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 1048576) to i64))
  %478 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %477, 0
  %479 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %478, ptr %477, 1
  %480 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %479, i64 0, 2
  %481 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %480, i64 1024, 3, 0
  %482 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %481, i64 1024, 3, 1
  %483 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %482, i64 1024, 4, 0
  %484 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %483, i64 1, 4, 1
  br label %485

485:                                              ; preds = %497, %476
  %486 = phi i64 [ %498, %497 ], [ 0, %476 ]
  %487 = icmp slt i64 %486, 1024
  br i1 %487, label %488, label %499

488:                                              ; preds = %491, %485
  %489 = phi i64 [ %496, %491 ], [ 0, %485 ]
  %490 = icmp slt i64 %489, 1024
  br i1 %490, label %491, label %497

491:                                              ; preds = %488
  %492 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %484, 1
  %493 = mul i64 %486, 1024
  %494 = add i64 %493, %489
  %495 = getelementptr double, ptr %492, i64 %494
  store double 0.000000e+00, ptr %495, align 8
  %496 = add i64 %489, 1
  br label %488

497:                                              ; preds = %488
  %498 = add i64 %486, 1
  br label %485

499:                                              ; preds = %533, %485
  %500 = phi i64 [ %534, %533 ], [ 0, %485 ]
  %501 = icmp slt i64 %500, 1024
  br i1 %501, label %502, label %535

502:                                              ; preds = %531, %499
  %503 = phi i64 [ %532, %531 ], [ 0, %499 ]
  %504 = icmp slt i64 %503, 1024
  br i1 %504, label %505, label %533

505:                                              ; preds = %508, %502
  %506 = phi i64 [ %530, %508 ], [ 0, %502 ]
  %507 = icmp slt i64 %506, 1024
  br i1 %507, label %508, label %531

508:                                              ; preds = %505
  %509 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %484, 1
  %510 = mul i64 %500, 1024
  %511 = add i64 %510, %503
  %512 = getelementptr double, ptr %509, i64 %511
  %513 = load double, ptr %512, align 8
  %514 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %436, 1
  %515 = mul i64 %500, 1024
  %516 = add i64 %515, %506
  %517 = getelementptr double, ptr %514, i64 %516
  %518 = load double, ptr %517, align 8
  %519 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %460, 1
  %520 = mul i64 %506, 1024
  %521 = add i64 %520, %503
  %522 = getelementptr double, ptr %519, i64 %521
  %523 = load double, ptr %522, align 8
  %524 = fmul double %518, %523
  %525 = fadd double %513, %524
  %526 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %484, 1
  %527 = mul i64 %500, 1024
  %528 = add i64 %527, %503
  %529 = getelementptr double, ptr %526, i64 %528
  store double %525, ptr %529, align 8
  %530 = add i64 %506, 1
  br label %505

531:                                              ; preds = %505
  %532 = add i64 %503, 1
  br label %502

533:                                              ; preds = %502
  %534 = add i64 %500, 1
  br label %499

535:                                              ; preds = %499
  call void @printFloat(double 1.000000e+01)
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
