	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0
	.p2align	4                               ## -- Begin function printMemrefF64
l_printMemrefF64:                       ## @printMemrefF64
	.cfi_startproc
## %bb.0:
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movq	%rsi, 16(%rsp)
	movq	%rdi, 8(%rsp)
	leaq	8(%rsp), %rdi
	callq	__mlir_ciface_printMemrefF64
	addq	$24, %rsp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3, 0x0                          ## -- Begin function main
LCPI1_0:
	.quad	0x4090000000000000              ## double 1024
LCPI1_1:
	.quad	0x3eb0000000000000              ## double 9.5367431640625E-7
LCPI1_2:
	.quad	0x4080000000000000              ## double 512
LCPI1_3:
	.quad	0x3ed0000000000000              ## double 3.814697265625E-6
LCPI1_4:
	.quad	0x4070000000000000              ## double 256
LCPI1_5:
	.quad	0x3ef0000000000000              ## double 1.52587890625E-5
LCPI1_6:
	.quad	0x4060000000000000              ## double 128
LCPI1_7:
	.quad	0x3f10000000000000              ## double 6.103515625E-5
LCPI1_8:
	.quad	0x4050000000000000              ## double 64
LCPI1_9:
	.quad	0x3f30000000000000              ## double 2.44140625E-4
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	4
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movl	$32768, %edi                    ## imm = 0x8000
	callq	_malloc
	movq	%rax, %rbx
	xorl	%eax, %eax
	vmovsd	LCPI1_9(%rip), %xmm3            ## xmm3 = [2.44140625E-4,0.0E+0]
	xorl	%ecx, %ecx
	jmp	LBB1_1
	.p2align	4
LBB1_11:                                ##   in Loop: Header=BB1_1 Depth=1
	addq	$32, %rcx
	addq	$16384, %rax                    ## imm = 0x4000
LBB1_1:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_3 Depth 2
                                        ##       Child Loop BB1_5 Depth 3
                                        ##         Child Loop BB1_8 Depth 4
	cmpq	$63, %rcx
	jg	LBB1_12
## %bb.2:                               ## %.preheader49
                                        ##   in Loop: Header=BB1_1 Depth=1
	movq	%rax, %rdx
	xorl	%esi, %esi
	jmp	LBB1_3
	.p2align	4
LBB1_10:                                ##   in Loop: Header=BB1_3 Depth=2
	addq	$32, %rsi
	addq	$256, %rdx                      ## imm = 0x100
LBB1_3:                                 ##   Parent Loop BB1_1 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_5 Depth 3
                                        ##         Child Loop BB1_8 Depth 4
	cmpq	$63, %rsi
	jg	LBB1_11
## %bb.4:                               ## %.preheader48
                                        ##   in Loop: Header=BB1_3 Depth=2
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	LBB1_5
	.p2align	4
LBB1_9:                                 ##   in Loop: Header=BB1_5 Depth=3
	incq	%r8
	addq	$512, %rdi                      ## imm = 0x200
LBB1_5:                                 ##   Parent Loop BB1_1 Depth=1
                                        ##     Parent Loop BB1_3 Depth=2
                                        ## =>    This Loop Header: Depth=3
                                        ##         Child Loop BB1_8 Depth 4
	cmpq	$31, %r8
	jg	LBB1_10
## %bb.6:                               ## %.preheader47
                                        ##   in Loop: Header=BB1_5 Depth=3
	leaq	(%rcx,%r8), %r9
	vcvtsi2sd	%r9, %xmm4, %xmm0
	leaq	(%rbx,%rdi), %r9
	xorl	%r10d, %r10d
	cmpq	$31, %r10
	jg	LBB1_9
	.p2align	4
LBB1_8:                                 ##   Parent Loop BB1_1 Depth=1
                                        ##     Parent Loop BB1_3 Depth=2
                                        ##       Parent Loop BB1_5 Depth=3
                                        ## =>      This Inner Loop Header: Depth=4
	leaq	(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm4, %xmm2
	vmulsd	LCPI1_8(%rip), %xmm0, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, (%r9,%r10,8)
	leaq	1(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm4, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 8(%r9,%r10,8)
	leaq	2(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm4, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 16(%r9,%r10,8)
	leaq	3(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm4, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 24(%r9,%r10,8)
	leaq	4(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm4, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 32(%r9,%r10,8)
	leaq	5(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm4, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 40(%r9,%r10,8)
	leaq	6(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm4, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 48(%r9,%r10,8)
	leaq	7(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm4, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm3, %xmm1, %xmm1
	vmovsd	%xmm1, 56(%r9,%r10,8)
	addq	$8, %r10
	cmpq	$31, %r10
	jle	LBB1_8
	jmp	LBB1_9
LBB1_12:
	movl	$32768, %edi                    ## imm = 0x8000
	callq	_malloc
	vmovsd	LCPI1_9(%rip), %xmm4            ## xmm4 = [2.44140625E-4,0.0E+0]
	vmovsd	LCPI1_8(%rip), %xmm3            ## xmm3 = [6.4E+1,0.0E+0]
	movq	%rax, %r14
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	jmp	LBB1_13
	.p2align	4
LBB1_23:                                ##   in Loop: Header=BB1_13 Depth=1
	addq	$32, %rcx
	addq	$16384, %rax                    ## imm = 0x4000
LBB1_13:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_15 Depth 2
                                        ##       Child Loop BB1_17 Depth 3
                                        ##         Child Loop BB1_20 Depth 4
	cmpq	$63, %rcx
	jg	LBB1_24
## %bb.14:                              ## %.preheader46
                                        ##   in Loop: Header=BB1_13 Depth=1
	movq	%rax, %rdx
	xorl	%esi, %esi
	jmp	LBB1_15
	.p2align	4
LBB1_22:                                ##   in Loop: Header=BB1_15 Depth=2
	addq	$32, %rsi
	addq	$256, %rdx                      ## imm = 0x100
LBB1_15:                                ##   Parent Loop BB1_13 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_17 Depth 3
                                        ##         Child Loop BB1_20 Depth 4
	cmpq	$63, %rsi
	jg	LBB1_23
## %bb.16:                              ## %.preheader45
                                        ##   in Loop: Header=BB1_15 Depth=2
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	LBB1_17
	.p2align	4
LBB1_21:                                ##   in Loop: Header=BB1_17 Depth=3
	incq	%r8
	addq	$512, %rdi                      ## imm = 0x200
LBB1_17:                                ##   Parent Loop BB1_13 Depth=1
                                        ##     Parent Loop BB1_15 Depth=2
                                        ## =>    This Loop Header: Depth=3
                                        ##         Child Loop BB1_20 Depth 4
	cmpq	$31, %r8
	jg	LBB1_22
## %bb.18:                              ## %.preheader44
                                        ##   in Loop: Header=BB1_17 Depth=3
	leaq	(%rcx,%r8), %r9
	vcvtsi2sd	%r9, %xmm5, %xmm0
	leaq	(%r14,%rdi), %r9
	xorl	%r10d, %r10d
	cmpq	$31, %r10
	jg	LBB1_21
	.p2align	4
LBB1_20:                                ##   Parent Loop BB1_13 Depth=1
                                        ##     Parent Loop BB1_15 Depth=2
                                        ##       Parent Loop BB1_17 Depth=3
                                        ## =>      This Inner Loop Header: Depth=4
	leaq	(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vmulsd	%xmm3, %xmm0, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, (%r9,%r10,8)
	leaq	1(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 8(%r9,%r10,8)
	leaq	2(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 16(%r9,%r10,8)
	leaq	3(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 24(%r9,%r10,8)
	leaq	4(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 32(%r9,%r10,8)
	leaq	5(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 40(%r9,%r10,8)
	leaq	6(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 48(%r9,%r10,8)
	leaq	7(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm4, %xmm1, %xmm1
	vmovsd	%xmm1, 56(%r9,%r10,8)
	addq	$8, %r10
	cmpq	$31, %r10
	jle	LBB1_20
	jmp	LBB1_21
LBB1_24:
	movl	$32768, %edi                    ## imm = 0x8000
	callq	_malloc
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB1_25
	.p2align	4
LBB1_29:                                ##   in Loop: Header=BB1_25 Depth=1
	incq	%rdx
	addq	$512, %rcx                      ## imm = 0x200
LBB1_25:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_28 Depth 2
	cmpq	$63, %rdx
	jg	LBB1_30
## %bb.26:                              ## %.preheader43
                                        ##   in Loop: Header=BB1_25 Depth=1
	leaq	(%rax,%rcx), %rsi
	xorl	%edi, %edi
	cmpq	$63, %rdi
	jg	LBB1_29
	.p2align	4
LBB1_28:                                ##   Parent Loop BB1_25 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	$0, (%rsi,%rdi,8)
	incq	%rdi
	cmpq	$63, %rdi
	jle	LBB1_28
	jmp	LBB1_29
LBB1_30:                                ## %.preheader42
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB1_31
	.p2align	4
LBB1_38:                                ##   in Loop: Header=BB1_31 Depth=1
	incq	%rdx
	addq	$512, %rcx                      ## imm = 0x200
LBB1_31:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_33 Depth 2
                                        ##       Child Loop BB1_36 Depth 3
	cmpq	$63, %rdx
	jg	LBB1_39
## %bb.32:                              ## %.preheader41
                                        ##   in Loop: Header=BB1_31 Depth=1
	movq	%rdx, %rsi
	shlq	$6, %rsi
	leaq	(%rbx,%rcx), %rdi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	jmp	LBB1_33
	.p2align	4
LBB1_37:                                ##   in Loop: Header=BB1_33 Depth=2
	incq	%r9
	addq	$8, %r8
LBB1_33:                                ##   Parent Loop BB1_31 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_36 Depth 3
	cmpq	$63, %r9
	jg	LBB1_38
## %bb.34:                              ## %.preheader40
                                        ##   in Loop: Header=BB1_33 Depth=2
	leaq	(%rsi,%r9), %r10
	movq	%r8, %r11
	xorl	%r15d, %r15d
	cmpq	$63, %r15
	jg	LBB1_37
	.p2align	4
LBB1_36:                                ##   Parent Loop BB1_31 Depth=1
                                        ##     Parent Loop BB1_33 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%r15,8), %xmm0            ## xmm0 = mem[0],zero
	vmovsd	(%r14,%r11), %xmm1              ## xmm1 = mem[0],zero
	vfmadd213sd	(%rax,%r10,8), %xmm0, %xmm1 ## xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%rax,%r10,8)
	incq	%r15
	addq	$512, %r11                      ## imm = 0x200
	cmpq	$63, %r15
	jle	LBB1_36
	jmp	LBB1_37
LBB1_39:
	movl	$131072, %edi                   ## imm = 0x20000
	callq	_malloc
	movq	%rax, %rbx
	xorl	%eax, %eax
	vmovsd	LCPI1_7(%rip), %xmm3            ## xmm3 = [6.103515625E-5,0.0E+0]
	xorl	%ecx, %ecx
	jmp	LBB1_40
	.p2align	4
LBB1_50:                                ##   in Loop: Header=BB1_40 Depth=1
	addq	$32, %rcx
	addq	$32768, %rax                    ## imm = 0x8000
LBB1_40:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_42 Depth 2
                                        ##       Child Loop BB1_44 Depth 3
                                        ##         Child Loop BB1_47 Depth 4
	cmpq	$127, %rcx
	jg	LBB1_51
## %bb.41:                              ## %.preheader39
                                        ##   in Loop: Header=BB1_40 Depth=1
	movq	%rax, %rdx
	xorl	%esi, %esi
	jmp	LBB1_42
	.p2align	4
LBB1_49:                                ##   in Loop: Header=BB1_42 Depth=2
	addq	$32, %rsi
	addq	$256, %rdx                      ## imm = 0x100
LBB1_42:                                ##   Parent Loop BB1_40 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_44 Depth 3
                                        ##         Child Loop BB1_47 Depth 4
	cmpq	$127, %rsi
	jg	LBB1_50
## %bb.43:                              ## %.preheader38
                                        ##   in Loop: Header=BB1_42 Depth=2
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	LBB1_44
	.p2align	4
LBB1_48:                                ##   in Loop: Header=BB1_44 Depth=3
	incq	%r8
	addq	$1024, %rdi                     ## imm = 0x400
LBB1_44:                                ##   Parent Loop BB1_40 Depth=1
                                        ##     Parent Loop BB1_42 Depth=2
                                        ## =>    This Loop Header: Depth=3
                                        ##         Child Loop BB1_47 Depth 4
	cmpq	$31, %r8
	jg	LBB1_49
## %bb.45:                              ## %.preheader37
                                        ##   in Loop: Header=BB1_44 Depth=3
	leaq	(%rcx,%r8), %r9
	vcvtsi2sd	%r9, %xmm5, %xmm0
	leaq	(%rbx,%rdi), %r9
	xorl	%r10d, %r10d
	cmpq	$31, %r10
	jg	LBB1_48
	.p2align	4
LBB1_47:                                ##   Parent Loop BB1_40 Depth=1
                                        ##     Parent Loop BB1_42 Depth=2
                                        ##       Parent Loop BB1_44 Depth=3
                                        ## =>      This Inner Loop Header: Depth=4
	leaq	(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vmulsd	LCPI1_6(%rip), %xmm0, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, (%r9,%r10,8)
	leaq	1(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 8(%r9,%r10,8)
	leaq	2(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 16(%r9,%r10,8)
	leaq	3(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 24(%r9,%r10,8)
	leaq	4(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 32(%r9,%r10,8)
	leaq	5(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 40(%r9,%r10,8)
	leaq	6(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 48(%r9,%r10,8)
	leaq	7(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm3, %xmm1, %xmm1
	vmovsd	%xmm1, 56(%r9,%r10,8)
	addq	$8, %r10
	cmpq	$31, %r10
	jle	LBB1_47
	jmp	LBB1_48
LBB1_51:
	movl	$131072, %edi                   ## imm = 0x20000
	callq	_malloc
	vmovsd	LCPI1_7(%rip), %xmm4            ## xmm4 = [6.103515625E-5,0.0E+0]
	vmovsd	LCPI1_6(%rip), %xmm3            ## xmm3 = [1.28E+2,0.0E+0]
	movq	%rax, %r14
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	jmp	LBB1_52
	.p2align	4
LBB1_62:                                ##   in Loop: Header=BB1_52 Depth=1
	addq	$32, %rcx
	addq	$32768, %rax                    ## imm = 0x8000
LBB1_52:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_54 Depth 2
                                        ##       Child Loop BB1_56 Depth 3
                                        ##         Child Loop BB1_59 Depth 4
	cmpq	$127, %rcx
	jg	LBB1_63
## %bb.53:                              ## %.preheader36
                                        ##   in Loop: Header=BB1_52 Depth=1
	movq	%rax, %rdx
	xorl	%esi, %esi
	jmp	LBB1_54
	.p2align	4
LBB1_61:                                ##   in Loop: Header=BB1_54 Depth=2
	addq	$32, %rsi
	addq	$256, %rdx                      ## imm = 0x100
LBB1_54:                                ##   Parent Loop BB1_52 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_56 Depth 3
                                        ##         Child Loop BB1_59 Depth 4
	cmpq	$127, %rsi
	jg	LBB1_62
## %bb.55:                              ## %.preheader35
                                        ##   in Loop: Header=BB1_54 Depth=2
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	LBB1_56
	.p2align	4
LBB1_60:                                ##   in Loop: Header=BB1_56 Depth=3
	incq	%r8
	addq	$1024, %rdi                     ## imm = 0x400
LBB1_56:                                ##   Parent Loop BB1_52 Depth=1
                                        ##     Parent Loop BB1_54 Depth=2
                                        ## =>    This Loop Header: Depth=3
                                        ##         Child Loop BB1_59 Depth 4
	cmpq	$31, %r8
	jg	LBB1_61
## %bb.57:                              ## %.preheader34
                                        ##   in Loop: Header=BB1_56 Depth=3
	leaq	(%rcx,%r8), %r9
	vcvtsi2sd	%r9, %xmm5, %xmm0
	leaq	(%r14,%rdi), %r9
	xorl	%r10d, %r10d
	cmpq	$31, %r10
	jg	LBB1_60
	.p2align	4
LBB1_59:                                ##   Parent Loop BB1_52 Depth=1
                                        ##     Parent Loop BB1_54 Depth=2
                                        ##       Parent Loop BB1_56 Depth=3
                                        ## =>      This Inner Loop Header: Depth=4
	leaq	(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vmulsd	%xmm3, %xmm0, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, (%r9,%r10,8)
	leaq	1(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 8(%r9,%r10,8)
	leaq	2(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 16(%r9,%r10,8)
	leaq	3(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 24(%r9,%r10,8)
	leaq	4(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 32(%r9,%r10,8)
	leaq	5(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 40(%r9,%r10,8)
	leaq	6(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 48(%r9,%r10,8)
	leaq	7(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm4, %xmm1, %xmm1
	vmovsd	%xmm1, 56(%r9,%r10,8)
	addq	$8, %r10
	cmpq	$31, %r10
	jle	LBB1_59
	jmp	LBB1_60
LBB1_63:
	movl	$131072, %edi                   ## imm = 0x20000
	callq	_malloc
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB1_64
	.p2align	4
LBB1_68:                                ##   in Loop: Header=BB1_64 Depth=1
	incq	%rdx
	addq	$1024, %rcx                     ## imm = 0x400
LBB1_64:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_67 Depth 2
	cmpq	$127, %rdx
	jg	LBB1_69
## %bb.65:                              ## %.preheader33
                                        ##   in Loop: Header=BB1_64 Depth=1
	leaq	(%rax,%rcx), %rsi
	xorl	%edi, %edi
	cmpq	$127, %rdi
	jg	LBB1_68
	.p2align	4
LBB1_67:                                ##   Parent Loop BB1_64 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	$0, (%rsi,%rdi,8)
	incq	%rdi
	cmpq	$127, %rdi
	jle	LBB1_67
	jmp	LBB1_68
LBB1_69:                                ## %.preheader32
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB1_70
	.p2align	4
LBB1_77:                                ##   in Loop: Header=BB1_70 Depth=1
	incq	%rdx
	addq	$1024, %rcx                     ## imm = 0x400
LBB1_70:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_72 Depth 2
                                        ##       Child Loop BB1_75 Depth 3
	cmpq	$127, %rdx
	jg	LBB1_78
## %bb.71:                              ## %.preheader31
                                        ##   in Loop: Header=BB1_70 Depth=1
	movq	%rdx, %rsi
	shlq	$7, %rsi
	leaq	(%rbx,%rcx), %rdi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	jmp	LBB1_72
	.p2align	4
LBB1_76:                                ##   in Loop: Header=BB1_72 Depth=2
	incq	%r9
	addq	$8, %r8
LBB1_72:                                ##   Parent Loop BB1_70 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_75 Depth 3
	cmpq	$127, %r9
	jg	LBB1_77
## %bb.73:                              ## %.preheader30
                                        ##   in Loop: Header=BB1_72 Depth=2
	leaq	(%rsi,%r9), %r10
	movq	%r8, %r11
	xorl	%r15d, %r15d
	cmpq	$127, %r15
	jg	LBB1_76
	.p2align	4
LBB1_75:                                ##   Parent Loop BB1_70 Depth=1
                                        ##     Parent Loop BB1_72 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%r15,8), %xmm0            ## xmm0 = mem[0],zero
	vmovsd	(%r14,%r11), %xmm1              ## xmm1 = mem[0],zero
	vfmadd213sd	(%rax,%r10,8), %xmm0, %xmm1 ## xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%rax,%r10,8)
	incq	%r15
	addq	$1024, %r11                     ## imm = 0x400
	cmpq	$127, %r15
	jle	LBB1_75
	jmp	LBB1_76
LBB1_78:
	movl	$524288, %edi                   ## imm = 0x80000
	callq	_malloc
	movq	%rax, %rbx
	xorl	%eax, %eax
	vmovsd	LCPI1_5(%rip), %xmm3            ## xmm3 = [1.52587890625E-5,0.0E+0]
	xorl	%ecx, %ecx
	jmp	LBB1_79
	.p2align	4
LBB1_89:                                ##   in Loop: Header=BB1_79 Depth=1
	addq	$32, %rcx
	addq	$65536, %rax                    ## imm = 0x10000
LBB1_79:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_81 Depth 2
                                        ##       Child Loop BB1_83 Depth 3
                                        ##         Child Loop BB1_86 Depth 4
	cmpq	$255, %rcx
	jg	LBB1_90
## %bb.80:                              ## %.preheader29
                                        ##   in Loop: Header=BB1_79 Depth=1
	movq	%rax, %rdx
	xorl	%esi, %esi
	jmp	LBB1_81
	.p2align	4
LBB1_88:                                ##   in Loop: Header=BB1_81 Depth=2
	addq	$32, %rsi
	addq	$256, %rdx                      ## imm = 0x100
LBB1_81:                                ##   Parent Loop BB1_79 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_83 Depth 3
                                        ##         Child Loop BB1_86 Depth 4
	cmpq	$255, %rsi
	jg	LBB1_89
## %bb.82:                              ## %.preheader28
                                        ##   in Loop: Header=BB1_81 Depth=2
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	LBB1_83
	.p2align	4
LBB1_87:                                ##   in Loop: Header=BB1_83 Depth=3
	incq	%r8
	addq	$2048, %rdi                     ## imm = 0x800
LBB1_83:                                ##   Parent Loop BB1_79 Depth=1
                                        ##     Parent Loop BB1_81 Depth=2
                                        ## =>    This Loop Header: Depth=3
                                        ##         Child Loop BB1_86 Depth 4
	cmpq	$31, %r8
	jg	LBB1_88
## %bb.84:                              ## %.preheader27
                                        ##   in Loop: Header=BB1_83 Depth=3
	leaq	(%rcx,%r8), %r9
	vcvtsi2sd	%r9, %xmm5, %xmm0
	leaq	(%rbx,%rdi), %r9
	xorl	%r10d, %r10d
	cmpq	$31, %r10
	jg	LBB1_87
	.p2align	4
LBB1_86:                                ##   Parent Loop BB1_79 Depth=1
                                        ##     Parent Loop BB1_81 Depth=2
                                        ##       Parent Loop BB1_83 Depth=3
                                        ## =>      This Inner Loop Header: Depth=4
	leaq	(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vmulsd	LCPI1_4(%rip), %xmm0, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, (%r9,%r10,8)
	leaq	1(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 8(%r9,%r10,8)
	leaq	2(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 16(%r9,%r10,8)
	leaq	3(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 24(%r9,%r10,8)
	leaq	4(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 32(%r9,%r10,8)
	leaq	5(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 40(%r9,%r10,8)
	leaq	6(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 48(%r9,%r10,8)
	leaq	7(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm3, %xmm1, %xmm1
	vmovsd	%xmm1, 56(%r9,%r10,8)
	addq	$8, %r10
	cmpq	$31, %r10
	jle	LBB1_86
	jmp	LBB1_87
LBB1_90:
	movl	$524288, %edi                   ## imm = 0x80000
	callq	_malloc
	vmovsd	LCPI1_5(%rip), %xmm4            ## xmm4 = [1.52587890625E-5,0.0E+0]
	vmovsd	LCPI1_4(%rip), %xmm3            ## xmm3 = [2.56E+2,0.0E+0]
	movq	%rax, %r14
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	jmp	LBB1_91
	.p2align	4
LBB1_101:                               ##   in Loop: Header=BB1_91 Depth=1
	addq	$32, %rcx
	addq	$65536, %rax                    ## imm = 0x10000
LBB1_91:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_93 Depth 2
                                        ##       Child Loop BB1_95 Depth 3
                                        ##         Child Loop BB1_98 Depth 4
	cmpq	$255, %rcx
	jg	LBB1_102
## %bb.92:                              ## %.preheader26
                                        ##   in Loop: Header=BB1_91 Depth=1
	movq	%rax, %rdx
	xorl	%esi, %esi
	jmp	LBB1_93
	.p2align	4
LBB1_100:                               ##   in Loop: Header=BB1_93 Depth=2
	addq	$32, %rsi
	addq	$256, %rdx                      ## imm = 0x100
LBB1_93:                                ##   Parent Loop BB1_91 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_95 Depth 3
                                        ##         Child Loop BB1_98 Depth 4
	cmpq	$255, %rsi
	jg	LBB1_101
## %bb.94:                              ## %.preheader25
                                        ##   in Loop: Header=BB1_93 Depth=2
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	LBB1_95
	.p2align	4
LBB1_99:                                ##   in Loop: Header=BB1_95 Depth=3
	incq	%r8
	addq	$2048, %rdi                     ## imm = 0x800
LBB1_95:                                ##   Parent Loop BB1_91 Depth=1
                                        ##     Parent Loop BB1_93 Depth=2
                                        ## =>    This Loop Header: Depth=3
                                        ##         Child Loop BB1_98 Depth 4
	cmpq	$31, %r8
	jg	LBB1_100
## %bb.96:                              ## %.preheader24
                                        ##   in Loop: Header=BB1_95 Depth=3
	leaq	(%rcx,%r8), %r9
	vcvtsi2sd	%r9, %xmm5, %xmm0
	leaq	(%r14,%rdi), %r9
	xorl	%r10d, %r10d
	cmpq	$31, %r10
	jg	LBB1_99
	.p2align	4
LBB1_98:                                ##   Parent Loop BB1_91 Depth=1
                                        ##     Parent Loop BB1_93 Depth=2
                                        ##       Parent Loop BB1_95 Depth=3
                                        ## =>      This Inner Loop Header: Depth=4
	leaq	(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vmulsd	%xmm3, %xmm0, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, (%r9,%r10,8)
	leaq	1(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 8(%r9,%r10,8)
	leaq	2(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 16(%r9,%r10,8)
	leaq	3(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 24(%r9,%r10,8)
	leaq	4(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 32(%r9,%r10,8)
	leaq	5(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 40(%r9,%r10,8)
	leaq	6(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 48(%r9,%r10,8)
	leaq	7(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm4, %xmm1, %xmm1
	vmovsd	%xmm1, 56(%r9,%r10,8)
	addq	$8, %r10
	cmpq	$31, %r10
	jle	LBB1_98
	jmp	LBB1_99
LBB1_102:
	movl	$524288, %edi                   ## imm = 0x80000
	callq	_malloc
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB1_103
	.p2align	4
LBB1_107:                               ##   in Loop: Header=BB1_103 Depth=1
	incq	%rdx
	addq	$2048, %rcx                     ## imm = 0x800
LBB1_103:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_106 Depth 2
	cmpq	$255, %rdx
	jg	LBB1_108
## %bb.104:                             ## %.preheader23
                                        ##   in Loop: Header=BB1_103 Depth=1
	leaq	(%rax,%rcx), %rsi
	xorl	%edi, %edi
	cmpq	$255, %rdi
	jg	LBB1_107
	.p2align	4
LBB1_106:                               ##   Parent Loop BB1_103 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	$0, (%rsi,%rdi,8)
	incq	%rdi
	cmpq	$255, %rdi
	jle	LBB1_106
	jmp	LBB1_107
LBB1_108:                               ## %.preheader22
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB1_109
	.p2align	4
LBB1_116:                               ##   in Loop: Header=BB1_109 Depth=1
	incq	%rdx
	addq	$2048, %rcx                     ## imm = 0x800
LBB1_109:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_111 Depth 2
                                        ##       Child Loop BB1_114 Depth 3
	cmpq	$255, %rdx
	jg	LBB1_117
## %bb.110:                             ## %.preheader21
                                        ##   in Loop: Header=BB1_109 Depth=1
	movq	%rdx, %rsi
	shlq	$8, %rsi
	leaq	(%rbx,%rcx), %rdi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	jmp	LBB1_111
	.p2align	4
LBB1_115:                               ##   in Loop: Header=BB1_111 Depth=2
	incq	%r9
	addq	$8, %r8
LBB1_111:                               ##   Parent Loop BB1_109 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_114 Depth 3
	cmpq	$255, %r9
	jg	LBB1_116
## %bb.112:                             ## %.preheader20
                                        ##   in Loop: Header=BB1_111 Depth=2
	leaq	(%rsi,%r9), %r10
	movq	%r8, %r11
	xorl	%r15d, %r15d
	cmpq	$255, %r15
	jg	LBB1_115
	.p2align	4
LBB1_114:                               ##   Parent Loop BB1_109 Depth=1
                                        ##     Parent Loop BB1_111 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%r15,8), %xmm0            ## xmm0 = mem[0],zero
	vmovsd	(%r14,%r11), %xmm1              ## xmm1 = mem[0],zero
	vfmadd213sd	(%rax,%r10,8), %xmm0, %xmm1 ## xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%rax,%r10,8)
	incq	%r15
	addq	$2048, %r11                     ## imm = 0x800
	cmpq	$255, %r15
	jle	LBB1_114
	jmp	LBB1_115
LBB1_117:
	movl	$2097152, %edi                  ## imm = 0x200000
	callq	_malloc
	movq	%rax, %rbx
	xorl	%eax, %eax
	vmovsd	LCPI1_3(%rip), %xmm3            ## xmm3 = [3.814697265625E-6,0.0E+0]
	xorl	%ecx, %ecx
	jmp	LBB1_118
	.p2align	4
LBB1_128:                               ##   in Loop: Header=BB1_118 Depth=1
	addq	$32, %rcx
	addq	$131072, %rax                   ## imm = 0x20000
LBB1_118:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_120 Depth 2
                                        ##       Child Loop BB1_122 Depth 3
                                        ##         Child Loop BB1_125 Depth 4
	cmpq	$511, %rcx                      ## imm = 0x1FF
	jg	LBB1_129
## %bb.119:                             ## %.preheader19
                                        ##   in Loop: Header=BB1_118 Depth=1
	movq	%rax, %rdx
	xorl	%esi, %esi
	jmp	LBB1_120
	.p2align	4
LBB1_127:                               ##   in Loop: Header=BB1_120 Depth=2
	addq	$32, %rsi
	addq	$256, %rdx                      ## imm = 0x100
LBB1_120:                               ##   Parent Loop BB1_118 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_122 Depth 3
                                        ##         Child Loop BB1_125 Depth 4
	cmpq	$511, %rsi                      ## imm = 0x1FF
	jg	LBB1_128
## %bb.121:                             ## %.preheader18
                                        ##   in Loop: Header=BB1_120 Depth=2
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	LBB1_122
	.p2align	4
LBB1_126:                               ##   in Loop: Header=BB1_122 Depth=3
	incq	%r8
	addq	$4096, %rdi                     ## imm = 0x1000
LBB1_122:                               ##   Parent Loop BB1_118 Depth=1
                                        ##     Parent Loop BB1_120 Depth=2
                                        ## =>    This Loop Header: Depth=3
                                        ##         Child Loop BB1_125 Depth 4
	cmpq	$31, %r8
	jg	LBB1_127
## %bb.123:                             ## %.preheader17
                                        ##   in Loop: Header=BB1_122 Depth=3
	leaq	(%rcx,%r8), %r9
	vcvtsi2sd	%r9, %xmm5, %xmm0
	leaq	(%rbx,%rdi), %r9
	xorl	%r10d, %r10d
	cmpq	$31, %r10
	jg	LBB1_126
	.p2align	4
LBB1_125:                               ##   Parent Loop BB1_118 Depth=1
                                        ##     Parent Loop BB1_120 Depth=2
                                        ##       Parent Loop BB1_122 Depth=3
                                        ## =>      This Inner Loop Header: Depth=4
	leaq	(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vmulsd	LCPI1_2(%rip), %xmm0, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, (%r9,%r10,8)
	leaq	1(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 8(%r9,%r10,8)
	leaq	2(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 16(%r9,%r10,8)
	leaq	3(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 24(%r9,%r10,8)
	leaq	4(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 32(%r9,%r10,8)
	leaq	5(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 40(%r9,%r10,8)
	leaq	6(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 48(%r9,%r10,8)
	leaq	7(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm3, %xmm1, %xmm1
	vmovsd	%xmm1, 56(%r9,%r10,8)
	addq	$8, %r10
	cmpq	$31, %r10
	jle	LBB1_125
	jmp	LBB1_126
LBB1_129:
	movl	$2097152, %edi                  ## imm = 0x200000
	callq	_malloc
	vmovsd	LCPI1_3(%rip), %xmm4            ## xmm4 = [3.814697265625E-6,0.0E+0]
	vmovsd	LCPI1_2(%rip), %xmm3            ## xmm3 = [5.12E+2,0.0E+0]
	movq	%rax, %r14
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	jmp	LBB1_130
	.p2align	4
LBB1_140:                               ##   in Loop: Header=BB1_130 Depth=1
	addq	$32, %rcx
	addq	$131072, %rax                   ## imm = 0x20000
LBB1_130:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_132 Depth 2
                                        ##       Child Loop BB1_134 Depth 3
                                        ##         Child Loop BB1_137 Depth 4
	cmpq	$511, %rcx                      ## imm = 0x1FF
	jg	LBB1_141
## %bb.131:                             ## %.preheader16
                                        ##   in Loop: Header=BB1_130 Depth=1
	movq	%rax, %rdx
	xorl	%esi, %esi
	jmp	LBB1_132
	.p2align	4
LBB1_139:                               ##   in Loop: Header=BB1_132 Depth=2
	addq	$32, %rsi
	addq	$256, %rdx                      ## imm = 0x100
LBB1_132:                               ##   Parent Loop BB1_130 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_134 Depth 3
                                        ##         Child Loop BB1_137 Depth 4
	cmpq	$511, %rsi                      ## imm = 0x1FF
	jg	LBB1_140
## %bb.133:                             ## %.preheader15
                                        ##   in Loop: Header=BB1_132 Depth=2
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	LBB1_134
	.p2align	4
LBB1_138:                               ##   in Loop: Header=BB1_134 Depth=3
	incq	%r8
	addq	$4096, %rdi                     ## imm = 0x1000
LBB1_134:                               ##   Parent Loop BB1_130 Depth=1
                                        ##     Parent Loop BB1_132 Depth=2
                                        ## =>    This Loop Header: Depth=3
                                        ##         Child Loop BB1_137 Depth 4
	cmpq	$31, %r8
	jg	LBB1_139
## %bb.135:                             ## %.preheader14
                                        ##   in Loop: Header=BB1_134 Depth=3
	leaq	(%rcx,%r8), %r9
	vcvtsi2sd	%r9, %xmm5, %xmm0
	leaq	(%r14,%rdi), %r9
	xorl	%r10d, %r10d
	cmpq	$31, %r10
	jg	LBB1_138
	.p2align	4
LBB1_137:                               ##   Parent Loop BB1_130 Depth=1
                                        ##     Parent Loop BB1_132 Depth=2
                                        ##       Parent Loop BB1_134 Depth=3
                                        ## =>      This Inner Loop Header: Depth=4
	leaq	(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vmulsd	%xmm3, %xmm0, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, (%r9,%r10,8)
	leaq	1(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 8(%r9,%r10,8)
	leaq	2(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 16(%r9,%r10,8)
	leaq	3(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 24(%r9,%r10,8)
	leaq	4(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 32(%r9,%r10,8)
	leaq	5(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 40(%r9,%r10,8)
	leaq	6(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 48(%r9,%r10,8)
	leaq	7(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm4, %xmm1, %xmm1
	vmovsd	%xmm1, 56(%r9,%r10,8)
	addq	$8, %r10
	cmpq	$31, %r10
	jle	LBB1_137
	jmp	LBB1_138
LBB1_141:
	movl	$2097152, %edi                  ## imm = 0x200000
	callq	_malloc
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB1_142
	.p2align	4
LBB1_146:                               ##   in Loop: Header=BB1_142 Depth=1
	incq	%rdx
	addq	$4096, %rcx                     ## imm = 0x1000
LBB1_142:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_145 Depth 2
	cmpq	$511, %rdx                      ## imm = 0x1FF
	jg	LBB1_147
## %bb.143:                             ## %.preheader13
                                        ##   in Loop: Header=BB1_142 Depth=1
	leaq	(%rax,%rcx), %rsi
	xorl	%edi, %edi
	cmpq	$511, %rdi                      ## imm = 0x1FF
	jg	LBB1_146
	.p2align	4
LBB1_145:                               ##   Parent Loop BB1_142 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	$0, (%rsi,%rdi,8)
	incq	%rdi
	cmpq	$511, %rdi                      ## imm = 0x1FF
	jle	LBB1_145
	jmp	LBB1_146
LBB1_147:                               ## %.preheader12
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB1_148
	.p2align	4
LBB1_155:                               ##   in Loop: Header=BB1_148 Depth=1
	incq	%rdx
	addq	$4096, %rcx                     ## imm = 0x1000
LBB1_148:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_150 Depth 2
                                        ##       Child Loop BB1_153 Depth 3
	cmpq	$511, %rdx                      ## imm = 0x1FF
	jg	LBB1_156
## %bb.149:                             ## %.preheader11
                                        ##   in Loop: Header=BB1_148 Depth=1
	movq	%rdx, %rsi
	shlq	$9, %rsi
	leaq	(%rbx,%rcx), %rdi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	jmp	LBB1_150
	.p2align	4
LBB1_154:                               ##   in Loop: Header=BB1_150 Depth=2
	incq	%r9
	addq	$8, %r8
LBB1_150:                               ##   Parent Loop BB1_148 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_153 Depth 3
	cmpq	$511, %r9                       ## imm = 0x1FF
	jg	LBB1_155
## %bb.151:                             ## %.preheader10
                                        ##   in Loop: Header=BB1_150 Depth=2
	leaq	(%rsi,%r9), %r10
	movq	%r8, %r11
	xorl	%r15d, %r15d
	cmpq	$511, %r15                      ## imm = 0x1FF
	jg	LBB1_154
	.p2align	4
LBB1_153:                               ##   Parent Loop BB1_148 Depth=1
                                        ##     Parent Loop BB1_150 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%r15,8), %xmm0            ## xmm0 = mem[0],zero
	vmovsd	(%r14,%r11), %xmm1              ## xmm1 = mem[0],zero
	vfmadd213sd	(%rax,%r10,8), %xmm0, %xmm1 ## xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%rax,%r10,8)
	incq	%r15
	addq	$4096, %r11                     ## imm = 0x1000
	cmpq	$511, %r15                      ## imm = 0x1FF
	jle	LBB1_153
	jmp	LBB1_154
LBB1_156:
	movl	$8388608, %edi                  ## imm = 0x800000
	callq	_malloc
	movq	%rax, %rbx
	xorl	%eax, %eax
	vmovsd	LCPI1_1(%rip), %xmm3            ## xmm3 = [9.5367431640625E-7,0.0E+0]
	xorl	%ecx, %ecx
	jmp	LBB1_157
	.p2align	4
LBB1_167:                               ##   in Loop: Header=BB1_157 Depth=1
	addq	$32, %rcx
	addq	$262144, %rax                   ## imm = 0x40000
LBB1_157:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_159 Depth 2
                                        ##       Child Loop BB1_161 Depth 3
                                        ##         Child Loop BB1_164 Depth 4
	cmpq	$1023, %rcx                     ## imm = 0x3FF
	jg	LBB1_168
## %bb.158:                             ## %.preheader9
                                        ##   in Loop: Header=BB1_157 Depth=1
	movq	%rax, %rdx
	xorl	%esi, %esi
	jmp	LBB1_159
	.p2align	4
LBB1_166:                               ##   in Loop: Header=BB1_159 Depth=2
	addq	$32, %rsi
	addq	$256, %rdx                      ## imm = 0x100
LBB1_159:                               ##   Parent Loop BB1_157 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_161 Depth 3
                                        ##         Child Loop BB1_164 Depth 4
	cmpq	$1023, %rsi                     ## imm = 0x3FF
	jg	LBB1_167
## %bb.160:                             ## %.preheader8
                                        ##   in Loop: Header=BB1_159 Depth=2
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	LBB1_161
	.p2align	4
LBB1_165:                               ##   in Loop: Header=BB1_161 Depth=3
	incq	%r8
	addq	$8192, %rdi                     ## imm = 0x2000
LBB1_161:                               ##   Parent Loop BB1_157 Depth=1
                                        ##     Parent Loop BB1_159 Depth=2
                                        ## =>    This Loop Header: Depth=3
                                        ##         Child Loop BB1_164 Depth 4
	cmpq	$31, %r8
	jg	LBB1_166
## %bb.162:                             ## %.preheader7
                                        ##   in Loop: Header=BB1_161 Depth=3
	leaq	(%rcx,%r8), %r9
	vcvtsi2sd	%r9, %xmm5, %xmm0
	leaq	(%rbx,%rdi), %r9
	xorl	%r10d, %r10d
	cmpq	$31, %r10
	jg	LBB1_165
	.p2align	4
LBB1_164:                               ##   Parent Loop BB1_157 Depth=1
                                        ##     Parent Loop BB1_159 Depth=2
                                        ##       Parent Loop BB1_161 Depth=3
                                        ## =>      This Inner Loop Header: Depth=4
	leaq	(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vmulsd	LCPI1_0(%rip), %xmm0, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, (%r9,%r10,8)
	leaq	1(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 8(%r9,%r10,8)
	leaq	2(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 16(%r9,%r10,8)
	leaq	3(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 24(%r9,%r10,8)
	leaq	4(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 32(%r9,%r10,8)
	leaq	5(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 40(%r9,%r10,8)
	leaq	6(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm3, %xmm2, %xmm2
	vmovsd	%xmm2, 48(%r9,%r10,8)
	leaq	7(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm3, %xmm1, %xmm1
	vmovsd	%xmm1, 56(%r9,%r10,8)
	addq	$8, %r10
	cmpq	$31, %r10
	jle	LBB1_164
	jmp	LBB1_165
LBB1_168:
	movl	$8388608, %edi                  ## imm = 0x800000
	callq	_malloc
	vmovsd	LCPI1_1(%rip), %xmm4            ## xmm4 = [9.5367431640625E-7,0.0E+0]
	vmovsd	LCPI1_0(%rip), %xmm3            ## xmm3 = [1.024E+3,0.0E+0]
	movq	%rax, %r14
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	jmp	LBB1_169
	.p2align	4
LBB1_179:                               ##   in Loop: Header=BB1_169 Depth=1
	addq	$32, %rcx
	addq	$262144, %rax                   ## imm = 0x40000
LBB1_169:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_171 Depth 2
                                        ##       Child Loop BB1_173 Depth 3
                                        ##         Child Loop BB1_176 Depth 4
	cmpq	$1023, %rcx                     ## imm = 0x3FF
	jg	LBB1_180
## %bb.170:                             ## %.preheader6
                                        ##   in Loop: Header=BB1_169 Depth=1
	movq	%rax, %rdx
	xorl	%esi, %esi
	jmp	LBB1_171
	.p2align	4
LBB1_178:                               ##   in Loop: Header=BB1_171 Depth=2
	addq	$32, %rsi
	addq	$256, %rdx                      ## imm = 0x100
LBB1_171:                               ##   Parent Loop BB1_169 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_173 Depth 3
                                        ##         Child Loop BB1_176 Depth 4
	cmpq	$1023, %rsi                     ## imm = 0x3FF
	jg	LBB1_179
## %bb.172:                             ## %.preheader5
                                        ##   in Loop: Header=BB1_171 Depth=2
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	LBB1_173
	.p2align	4
LBB1_177:                               ##   in Loop: Header=BB1_173 Depth=3
	incq	%r8
	addq	$8192, %rdi                     ## imm = 0x2000
LBB1_173:                               ##   Parent Loop BB1_169 Depth=1
                                        ##     Parent Loop BB1_171 Depth=2
                                        ## =>    This Loop Header: Depth=3
                                        ##         Child Loop BB1_176 Depth 4
	cmpq	$31, %r8
	jg	LBB1_178
## %bb.174:                             ## %.preheader4
                                        ##   in Loop: Header=BB1_173 Depth=3
	leaq	(%rcx,%r8), %r9
	vcvtsi2sd	%r9, %xmm5, %xmm0
	leaq	(%r14,%rdi), %r9
	xorl	%r10d, %r10d
	cmpq	$31, %r10
	jg	LBB1_177
	.p2align	4
LBB1_176:                               ##   Parent Loop BB1_169 Depth=1
                                        ##     Parent Loop BB1_171 Depth=2
                                        ##       Parent Loop BB1_173 Depth=3
                                        ## =>      This Inner Loop Header: Depth=4
	leaq	(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vmulsd	%xmm3, %xmm0, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, (%r9,%r10,8)
	leaq	1(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 8(%r9,%r10,8)
	leaq	2(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 16(%r9,%r10,8)
	leaq	3(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 24(%r9,%r10,8)
	leaq	4(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 32(%r9,%r10,8)
	leaq	5(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 40(%r9,%r10,8)
	leaq	6(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm2
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	%xmm2, 48(%r9,%r10,8)
	leaq	7(%rsi,%r10), %r11
	vcvtsi2sd	%r11, %xmm5, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm4, %xmm1, %xmm1
	vmovsd	%xmm1, 56(%r9,%r10,8)
	addq	$8, %r10
	cmpq	$31, %r10
	jle	LBB1_176
	jmp	LBB1_177
LBB1_180:
	movl	$8388608, %edi                  ## imm = 0x800000
	callq	_malloc
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB1_181
	.p2align	4
LBB1_185:                               ##   in Loop: Header=BB1_181 Depth=1
	incq	%rdx
	addq	$8192, %rcx                     ## imm = 0x2000
LBB1_181:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_184 Depth 2
	cmpq	$1023, %rdx                     ## imm = 0x3FF
	jg	LBB1_186
## %bb.182:                             ## %.preheader3
                                        ##   in Loop: Header=BB1_181 Depth=1
	leaq	(%rax,%rcx), %rsi
	xorl	%edi, %edi
	cmpq	$1023, %rdi                     ## imm = 0x3FF
	jg	LBB1_185
	.p2align	4
LBB1_184:                               ##   Parent Loop BB1_181 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	$0, (%rsi,%rdi,8)
	incq	%rdi
	cmpq	$1023, %rdi                     ## imm = 0x3FF
	jle	LBB1_184
	jmp	LBB1_185
LBB1_186:                               ## %.preheader2
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB1_187
	.p2align	4
LBB1_194:                               ##   in Loop: Header=BB1_187 Depth=1
	incq	%rdx
	addq	$8192, %rcx                     ## imm = 0x2000
LBB1_187:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB1_189 Depth 2
                                        ##       Child Loop BB1_192 Depth 3
	cmpq	$1023, %rdx                     ## imm = 0x3FF
	jg	LBB1_195
## %bb.188:                             ## %.preheader1
                                        ##   in Loop: Header=BB1_187 Depth=1
	movq	%rdx, %rsi
	shlq	$10, %rsi
	leaq	(%rbx,%rcx), %rdi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	jmp	LBB1_189
	.p2align	4
LBB1_193:                               ##   in Loop: Header=BB1_189 Depth=2
	incq	%r9
	addq	$8, %r8
LBB1_189:                               ##   Parent Loop BB1_187 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB1_192 Depth 3
	cmpq	$1023, %r9                      ## imm = 0x3FF
	jg	LBB1_194
## %bb.190:                             ## %.preheader
                                        ##   in Loop: Header=BB1_189 Depth=2
	leaq	(%rsi,%r9), %r10
	movq	%r8, %r11
	xorl	%r15d, %r15d
	cmpq	$1023, %r15                     ## imm = 0x3FF
	jg	LBB1_193
	.p2align	4
LBB1_192:                               ##   Parent Loop BB1_187 Depth=1
                                        ##     Parent Loop BB1_189 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%r15,8), %xmm0            ## xmm0 = mem[0],zero
	vmovsd	(%r14,%r11), %xmm1              ## xmm1 = mem[0],zero
	vfmadd213sd	(%rax,%r10,8), %xmm0, %xmm1 ## xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%rax,%r10,8)
	incq	%r15
	addq	$8192, %r11                     ## imm = 0x2000
	cmpq	$1023, %r15                     ## imm = 0x3FF
	jle	LBB1_192
	jmp	LBB1_193
LBB1_195:
	xorl	%eax, %eax
	popq	%rbx
	popq	%r14
	popq	%r15
	retq
	.cfi_endproc
                                        ## -- End function
.subsections_via_symbols
