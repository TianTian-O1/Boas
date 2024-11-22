	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3, 0x0                          ## -- Begin function main
LCPI0_0:
	.quad	0x3ff0000000000000              ## double 1
LCPI0_1:
	.quad	0x4000000000000000              ## double 2
LCPI0_2:
	.quad	0x4008000000000000              ## double 3
LCPI0_3:
	.quad	0x4010000000000000              ## double 4
LCPI0_4:
	.quad	0x4014000000000000              ## double 5
LCPI0_5:
	.quad	0x4018000000000000              ## double 6
LCPI0_6:
	.quad	0x401c000000000000              ## double 7
LCPI0_7:
	.quad	0x4020000000000000              ## double 8
LCPI0_8:
	.quad	0x4022000000000000              ## double 9
LCPI0_9:
	.quad	0x4024000000000000              ## double 10
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	4
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	vmovsd	LCPI0_0(%rip), %xmm0            ## xmm0 = [1.0E+0,0.0E+0]
	callq	_printFloat
	movl	$32768, %edi                    ## imm = 0x8000
	callq	_malloc
	movq	%rax, %rbx
	xorl	%r14d, %r14d
	xorl	%r15d, %r15d
	jmp	LBB0_1
	.p2align	4
LBB0_5:                                 ##   in Loop: Header=BB0_1 Depth=1
	incq	%r15
	addq	$512, %r14                      ## imm = 0x200
LBB0_1:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_4 Depth 2
	cmpq	$63, %r15
	jg	LBB0_6
## %bb.2:                               ## %.preheader29
                                        ##   in Loop: Header=BB0_1 Depth=1
	leaq	(%rbx,%r14), %r12
	xorl	%r13d, %r13d
	cmpq	$63, %r13
	jg	LBB0_5
	.p2align	4
LBB0_4:                                 ##   Parent Loop BB0_1 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	vmovsd	%xmm0, (%r12,%r13,8)
	incq	%r13
	cmpq	$63, %r13
	jle	LBB0_4
	jmp	LBB0_5
LBB0_6:
	movl	$32768, %edi                    ## imm = 0x8000
	callq	_malloc
	movq	%rax, %r14
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	jmp	LBB0_7
	.p2align	4
LBB0_11:                                ##   in Loop: Header=BB0_7 Depth=1
	incq	%r12
	addq	$512, %r15                      ## imm = 0x200
LBB0_7:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_10 Depth 2
	cmpq	$63, %r12
	jg	LBB0_12
## %bb.8:                               ## %.preheader28
                                        ##   in Loop: Header=BB0_7 Depth=1
	leaq	(%r14,%r15), %r13
	xorl	%ebp, %ebp
	cmpq	$63, %rbp
	jg	LBB0_11
	.p2align	4
LBB0_10:                                ##   Parent Loop BB0_7 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	vmovsd	%xmm0, (%r13,%rbp,8)
	incq	%rbp
	cmpq	$63, %rbp
	jle	LBB0_10
	jmp	LBB0_11
LBB0_12:
	movl	$32768, %edi                    ## imm = 0x8000
	callq	_malloc
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB0_13
	.p2align	4
LBB0_17:                                ##   in Loop: Header=BB0_13 Depth=1
	incq	%rdx
	addq	$512, %rcx                      ## imm = 0x200
LBB0_13:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_16 Depth 2
	cmpq	$63, %rdx
	jg	LBB0_18
## %bb.14:                              ## %.preheader27
                                        ##   in Loop: Header=BB0_13 Depth=1
	leaq	(%rax,%rcx), %rsi
	xorl	%edi, %edi
	cmpq	$63, %rdi
	jg	LBB0_17
	.p2align	4
LBB0_16:                                ##   Parent Loop BB0_13 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	$0, (%rsi,%rdi,8)
	incq	%rdi
	cmpq	$63, %rdi
	jle	LBB0_16
	jmp	LBB0_17
LBB0_18:                                ## %.preheader26
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB0_19
	.p2align	4
LBB0_26:                                ##   in Loop: Header=BB0_19 Depth=1
	incq	%rdx
	addq	$512, %rcx                      ## imm = 0x200
LBB0_19:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_21 Depth 2
                                        ##       Child Loop BB0_24 Depth 3
	cmpq	$63, %rdx
	jg	LBB0_27
## %bb.20:                              ## %.preheader25
                                        ##   in Loop: Header=BB0_19 Depth=1
	movq	%rdx, %rsi
	shlq	$6, %rsi
	leaq	(%rbx,%rcx), %rdi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	jmp	LBB0_21
	.p2align	4
LBB0_25:                                ##   in Loop: Header=BB0_21 Depth=2
	incq	%r9
	addq	$8, %r8
LBB0_21:                                ##   Parent Loop BB0_19 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_24 Depth 3
	cmpq	$63, %r9
	jg	LBB0_26
## %bb.22:                              ## %.preheader24
                                        ##   in Loop: Header=BB0_21 Depth=2
	leaq	(%rsi,%r9), %r10
	movq	%r8, %r11
	xorl	%r15d, %r15d
	cmpq	$63, %r15
	jg	LBB0_25
	.p2align	4
LBB0_24:                                ##   Parent Loop BB0_19 Depth=1
                                        ##     Parent Loop BB0_21 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%r15,8), %xmm0            ## xmm0 = mem[0],zero
	vmovsd	(%r14,%r11), %xmm1              ## xmm1 = mem[0],zero
	vfmadd213sd	(%rax,%r10,8), %xmm0, %xmm1 ## xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%rax,%r10,8)
	incq	%r15
	addq	$512, %r11                      ## imm = 0x200
	cmpq	$63, %r15
	jle	LBB0_24
	jmp	LBB0_25
LBB0_27:
	vmovsd	LCPI0_1(%rip), %xmm0            ## xmm0 = [2.0E+0,0.0E+0]
	callq	_printFloat
	vmovsd	LCPI0_2(%rip), %xmm0            ## xmm0 = [3.0E+0,0.0E+0]
	callq	_printFloat
	movl	$131072, %edi                   ## imm = 0x20000
	callq	_malloc
	movq	%rax, %rbx
	xorl	%r14d, %r14d
	xorl	%r15d, %r15d
	jmp	LBB0_28
	.p2align	4
LBB0_32:                                ##   in Loop: Header=BB0_28 Depth=1
	incq	%r15
	addq	$1024, %r14                     ## imm = 0x400
LBB0_28:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_31 Depth 2
	cmpq	$127, %r15
	jg	LBB0_33
## %bb.29:                              ## %.preheader23
                                        ##   in Loop: Header=BB0_28 Depth=1
	leaq	(%rbx,%r14), %r12
	xorl	%r13d, %r13d
	cmpq	$127, %r13
	jg	LBB0_32
	.p2align	4
LBB0_31:                                ##   Parent Loop BB0_28 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	vmovsd	%xmm0, (%r12,%r13,8)
	incq	%r13
	cmpq	$127, %r13
	jle	LBB0_31
	jmp	LBB0_32
LBB0_33:
	movl	$131072, %edi                   ## imm = 0x20000
	callq	_malloc
	movq	%rax, %r14
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	jmp	LBB0_34
	.p2align	4
LBB0_38:                                ##   in Loop: Header=BB0_34 Depth=1
	incq	%r12
	addq	$1024, %r15                     ## imm = 0x400
LBB0_34:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_37 Depth 2
	cmpq	$127, %r12
	jg	LBB0_39
## %bb.35:                              ## %.preheader22
                                        ##   in Loop: Header=BB0_34 Depth=1
	leaq	(%r14,%r15), %r13
	xorl	%ebp, %ebp
	cmpq	$127, %rbp
	jg	LBB0_38
	.p2align	4
LBB0_37:                                ##   Parent Loop BB0_34 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	vmovsd	%xmm0, (%r13,%rbp,8)
	incq	%rbp
	cmpq	$127, %rbp
	jle	LBB0_37
	jmp	LBB0_38
LBB0_39:
	movl	$131072, %edi                   ## imm = 0x20000
	callq	_malloc
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB0_40
	.p2align	4
LBB0_44:                                ##   in Loop: Header=BB0_40 Depth=1
	incq	%rdx
	addq	$1024, %rcx                     ## imm = 0x400
LBB0_40:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_43 Depth 2
	cmpq	$127, %rdx
	jg	LBB0_45
## %bb.41:                              ## %.preheader21
                                        ##   in Loop: Header=BB0_40 Depth=1
	leaq	(%rax,%rcx), %rsi
	xorl	%edi, %edi
	cmpq	$127, %rdi
	jg	LBB0_44
	.p2align	4
LBB0_43:                                ##   Parent Loop BB0_40 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	$0, (%rsi,%rdi,8)
	incq	%rdi
	cmpq	$127, %rdi
	jle	LBB0_43
	jmp	LBB0_44
LBB0_45:                                ## %.preheader20
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB0_46
	.p2align	4
LBB0_53:                                ##   in Loop: Header=BB0_46 Depth=1
	incq	%rdx
	addq	$1024, %rcx                     ## imm = 0x400
LBB0_46:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_48 Depth 2
                                        ##       Child Loop BB0_51 Depth 3
	cmpq	$127, %rdx
	jg	LBB0_54
## %bb.47:                              ## %.preheader19
                                        ##   in Loop: Header=BB0_46 Depth=1
	movq	%rdx, %rsi
	shlq	$7, %rsi
	leaq	(%rbx,%rcx), %rdi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	jmp	LBB0_48
	.p2align	4
LBB0_52:                                ##   in Loop: Header=BB0_48 Depth=2
	incq	%r9
	addq	$8, %r8
LBB0_48:                                ##   Parent Loop BB0_46 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_51 Depth 3
	cmpq	$127, %r9
	jg	LBB0_53
## %bb.49:                              ## %.preheader18
                                        ##   in Loop: Header=BB0_48 Depth=2
	leaq	(%rsi,%r9), %r10
	movq	%r8, %r11
	xorl	%r15d, %r15d
	cmpq	$127, %r15
	jg	LBB0_52
	.p2align	4
LBB0_51:                                ##   Parent Loop BB0_46 Depth=1
                                        ##     Parent Loop BB0_48 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%r15,8), %xmm0            ## xmm0 = mem[0],zero
	vmovsd	(%r14,%r11), %xmm1              ## xmm1 = mem[0],zero
	vfmadd213sd	(%rax,%r10,8), %xmm0, %xmm1 ## xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%rax,%r10,8)
	incq	%r15
	addq	$1024, %r11                     ## imm = 0x400
	cmpq	$127, %r15
	jle	LBB0_51
	jmp	LBB0_52
LBB0_54:
	vmovsd	LCPI0_3(%rip), %xmm0            ## xmm0 = [4.0E+0,0.0E+0]
	callq	_printFloat
	vmovsd	LCPI0_4(%rip), %xmm0            ## xmm0 = [5.0E+0,0.0E+0]
	callq	_printFloat
	movl	$524288, %edi                   ## imm = 0x80000
	callq	_malloc
	movq	%rax, %rbx
	xorl	%r14d, %r14d
	xorl	%r15d, %r15d
	jmp	LBB0_55
	.p2align	4
LBB0_59:                                ##   in Loop: Header=BB0_55 Depth=1
	incq	%r15
	addq	$2048, %r14                     ## imm = 0x800
LBB0_55:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_58 Depth 2
	cmpq	$255, %r15
	jg	LBB0_60
## %bb.56:                              ## %.preheader17
                                        ##   in Loop: Header=BB0_55 Depth=1
	leaq	(%rbx,%r14), %r12
	xorl	%r13d, %r13d
	cmpq	$255, %r13
	jg	LBB0_59
	.p2align	4
LBB0_58:                                ##   Parent Loop BB0_55 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	vmovsd	%xmm0, (%r12,%r13,8)
	incq	%r13
	cmpq	$255, %r13
	jle	LBB0_58
	jmp	LBB0_59
LBB0_60:
	movl	$524288, %edi                   ## imm = 0x80000
	callq	_malloc
	movq	%rax, %r14
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	jmp	LBB0_61
	.p2align	4
LBB0_65:                                ##   in Loop: Header=BB0_61 Depth=1
	incq	%r12
	addq	$2048, %r15                     ## imm = 0x800
LBB0_61:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_64 Depth 2
	cmpq	$255, %r12
	jg	LBB0_66
## %bb.62:                              ## %.preheader16
                                        ##   in Loop: Header=BB0_61 Depth=1
	leaq	(%r14,%r15), %r13
	xorl	%ebp, %ebp
	cmpq	$255, %rbp
	jg	LBB0_65
	.p2align	4
LBB0_64:                                ##   Parent Loop BB0_61 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	vmovsd	%xmm0, (%r13,%rbp,8)
	incq	%rbp
	cmpq	$255, %rbp
	jle	LBB0_64
	jmp	LBB0_65
LBB0_66:
	movl	$524288, %edi                   ## imm = 0x80000
	callq	_malloc
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB0_67
	.p2align	4
LBB0_71:                                ##   in Loop: Header=BB0_67 Depth=1
	incq	%rdx
	addq	$2048, %rcx                     ## imm = 0x800
LBB0_67:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_70 Depth 2
	cmpq	$255, %rdx
	jg	LBB0_72
## %bb.68:                              ## %.preheader15
                                        ##   in Loop: Header=BB0_67 Depth=1
	leaq	(%rax,%rcx), %rsi
	xorl	%edi, %edi
	cmpq	$255, %rdi
	jg	LBB0_71
	.p2align	4
LBB0_70:                                ##   Parent Loop BB0_67 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	$0, (%rsi,%rdi,8)
	incq	%rdi
	cmpq	$255, %rdi
	jle	LBB0_70
	jmp	LBB0_71
LBB0_72:                                ## %.preheader14
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB0_73
	.p2align	4
LBB0_80:                                ##   in Loop: Header=BB0_73 Depth=1
	incq	%rdx
	addq	$2048, %rcx                     ## imm = 0x800
LBB0_73:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_75 Depth 2
                                        ##       Child Loop BB0_78 Depth 3
	cmpq	$255, %rdx
	jg	LBB0_81
## %bb.74:                              ## %.preheader13
                                        ##   in Loop: Header=BB0_73 Depth=1
	movq	%rdx, %rsi
	shlq	$8, %rsi
	leaq	(%rbx,%rcx), %rdi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	jmp	LBB0_75
	.p2align	4
LBB0_79:                                ##   in Loop: Header=BB0_75 Depth=2
	incq	%r9
	addq	$8, %r8
LBB0_75:                                ##   Parent Loop BB0_73 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_78 Depth 3
	cmpq	$255, %r9
	jg	LBB0_80
## %bb.76:                              ## %.preheader12
                                        ##   in Loop: Header=BB0_75 Depth=2
	leaq	(%rsi,%r9), %r10
	movq	%r8, %r11
	xorl	%r15d, %r15d
	cmpq	$255, %r15
	jg	LBB0_79
	.p2align	4
LBB0_78:                                ##   Parent Loop BB0_73 Depth=1
                                        ##     Parent Loop BB0_75 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%r15,8), %xmm0            ## xmm0 = mem[0],zero
	vmovsd	(%r14,%r11), %xmm1              ## xmm1 = mem[0],zero
	vfmadd213sd	(%rax,%r10,8), %xmm0, %xmm1 ## xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%rax,%r10,8)
	incq	%r15
	addq	$2048, %r11                     ## imm = 0x800
	cmpq	$255, %r15
	jle	LBB0_78
	jmp	LBB0_79
LBB0_81:
	vmovsd	LCPI0_5(%rip), %xmm0            ## xmm0 = [6.0E+0,0.0E+0]
	callq	_printFloat
	vmovsd	LCPI0_6(%rip), %xmm0            ## xmm0 = [7.0E+0,0.0E+0]
	callq	_printFloat
	movl	$2097152, %edi                  ## imm = 0x200000
	callq	_malloc
	movq	%rax, %rbx
	xorl	%r14d, %r14d
	xorl	%r15d, %r15d
	jmp	LBB0_82
	.p2align	4
LBB0_86:                                ##   in Loop: Header=BB0_82 Depth=1
	incq	%r15
	addq	$4096, %r14                     ## imm = 0x1000
LBB0_82:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_85 Depth 2
	cmpq	$511, %r15                      ## imm = 0x1FF
	jg	LBB0_87
## %bb.83:                              ## %.preheader11
                                        ##   in Loop: Header=BB0_82 Depth=1
	leaq	(%rbx,%r14), %r12
	xorl	%r13d, %r13d
	cmpq	$511, %r13                      ## imm = 0x1FF
	jg	LBB0_86
	.p2align	4
LBB0_85:                                ##   Parent Loop BB0_82 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	vmovsd	%xmm0, (%r12,%r13,8)
	incq	%r13
	cmpq	$511, %r13                      ## imm = 0x1FF
	jle	LBB0_85
	jmp	LBB0_86
LBB0_87:
	movl	$2097152, %edi                  ## imm = 0x200000
	callq	_malloc
	movq	%rax, %r14
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	jmp	LBB0_88
	.p2align	4
LBB0_92:                                ##   in Loop: Header=BB0_88 Depth=1
	incq	%r12
	addq	$4096, %r15                     ## imm = 0x1000
LBB0_88:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_91 Depth 2
	cmpq	$511, %r12                      ## imm = 0x1FF
	jg	LBB0_93
## %bb.89:                              ## %.preheader10
                                        ##   in Loop: Header=BB0_88 Depth=1
	leaq	(%r14,%r15), %r13
	xorl	%ebp, %ebp
	cmpq	$511, %rbp                      ## imm = 0x1FF
	jg	LBB0_92
	.p2align	4
LBB0_91:                                ##   Parent Loop BB0_88 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	vmovsd	%xmm0, (%r13,%rbp,8)
	incq	%rbp
	cmpq	$511, %rbp                      ## imm = 0x1FF
	jle	LBB0_91
	jmp	LBB0_92
LBB0_93:
	movl	$2097152, %edi                  ## imm = 0x200000
	callq	_malloc
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB0_94
	.p2align	4
LBB0_98:                                ##   in Loop: Header=BB0_94 Depth=1
	incq	%rdx
	addq	$4096, %rcx                     ## imm = 0x1000
LBB0_94:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_97 Depth 2
	cmpq	$511, %rdx                      ## imm = 0x1FF
	jg	LBB0_99
## %bb.95:                              ## %.preheader9
                                        ##   in Loop: Header=BB0_94 Depth=1
	leaq	(%rax,%rcx), %rsi
	xorl	%edi, %edi
	cmpq	$511, %rdi                      ## imm = 0x1FF
	jg	LBB0_98
	.p2align	4
LBB0_97:                                ##   Parent Loop BB0_94 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	$0, (%rsi,%rdi,8)
	incq	%rdi
	cmpq	$511, %rdi                      ## imm = 0x1FF
	jle	LBB0_97
	jmp	LBB0_98
LBB0_99:                                ## %.preheader8
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB0_100
	.p2align	4
LBB0_107:                               ##   in Loop: Header=BB0_100 Depth=1
	incq	%rdx
	addq	$4096, %rcx                     ## imm = 0x1000
LBB0_100:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_102 Depth 2
                                        ##       Child Loop BB0_105 Depth 3
	cmpq	$511, %rdx                      ## imm = 0x1FF
	jg	LBB0_108
## %bb.101:                             ## %.preheader7
                                        ##   in Loop: Header=BB0_100 Depth=1
	movq	%rdx, %rsi
	shlq	$9, %rsi
	leaq	(%rbx,%rcx), %rdi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	jmp	LBB0_102
	.p2align	4
LBB0_106:                               ##   in Loop: Header=BB0_102 Depth=2
	incq	%r9
	addq	$8, %r8
LBB0_102:                               ##   Parent Loop BB0_100 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_105 Depth 3
	cmpq	$511, %r9                       ## imm = 0x1FF
	jg	LBB0_107
## %bb.103:                             ## %.preheader6
                                        ##   in Loop: Header=BB0_102 Depth=2
	leaq	(%rsi,%r9), %r10
	movq	%r8, %r11
	xorl	%r15d, %r15d
	cmpq	$511, %r15                      ## imm = 0x1FF
	jg	LBB0_106
	.p2align	4
LBB0_105:                               ##   Parent Loop BB0_100 Depth=1
                                        ##     Parent Loop BB0_102 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%r15,8), %xmm0            ## xmm0 = mem[0],zero
	vmovsd	(%r14,%r11), %xmm1              ## xmm1 = mem[0],zero
	vfmadd213sd	(%rax,%r10,8), %xmm0, %xmm1 ## xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%rax,%r10,8)
	incq	%r15
	addq	$4096, %r11                     ## imm = 0x1000
	cmpq	$511, %r15                      ## imm = 0x1FF
	jle	LBB0_105
	jmp	LBB0_106
LBB0_108:
	vmovsd	LCPI0_7(%rip), %xmm0            ## xmm0 = [8.0E+0,0.0E+0]
	callq	_printFloat
	vmovsd	LCPI0_8(%rip), %xmm0            ## xmm0 = [9.0E+0,0.0E+0]
	callq	_printFloat
	movl	$8388608, %edi                  ## imm = 0x800000
	callq	_malloc
	movq	%rax, %rbx
	xorl	%r14d, %r14d
	xorl	%r15d, %r15d
	jmp	LBB0_109
	.p2align	4
LBB0_113:                               ##   in Loop: Header=BB0_109 Depth=1
	incq	%r15
	addq	$8192, %r14                     ## imm = 0x2000
LBB0_109:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_112 Depth 2
	cmpq	$1023, %r15                     ## imm = 0x3FF
	jg	LBB0_114
## %bb.110:                             ## %.preheader5
                                        ##   in Loop: Header=BB0_109 Depth=1
	leaq	(%rbx,%r14), %r12
	xorl	%r13d, %r13d
	cmpq	$1023, %r13                     ## imm = 0x3FF
	jg	LBB0_113
	.p2align	4
LBB0_112:                               ##   Parent Loop BB0_109 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	vmovsd	%xmm0, (%r12,%r13,8)
	incq	%r13
	cmpq	$1023, %r13                     ## imm = 0x3FF
	jle	LBB0_112
	jmp	LBB0_113
LBB0_114:
	movl	$8388608, %edi                  ## imm = 0x800000
	callq	_malloc
	movq	%rax, %r14
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	jmp	LBB0_115
	.p2align	4
LBB0_119:                               ##   in Loop: Header=BB0_115 Depth=1
	incq	%r12
	addq	$8192, %r15                     ## imm = 0x2000
LBB0_115:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_118 Depth 2
	cmpq	$1023, %r12                     ## imm = 0x3FF
	jg	LBB0_120
## %bb.116:                             ## %.preheader4
                                        ##   in Loop: Header=BB0_115 Depth=1
	leaq	(%r14,%r15), %r13
	xorl	%ebp, %ebp
	cmpq	$1023, %rbp                     ## imm = 0x3FF
	jg	LBB0_119
	.p2align	4
LBB0_118:                               ##   Parent Loop BB0_115 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	callq	_generate_random
	vmovsd	%xmm0, (%r13,%rbp,8)
	incq	%rbp
	cmpq	$1023, %rbp                     ## imm = 0x3FF
	jle	LBB0_118
	jmp	LBB0_119
LBB0_120:
	movl	$8388608, %edi                  ## imm = 0x800000
	callq	_malloc
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB0_121
	.p2align	4
LBB0_125:                               ##   in Loop: Header=BB0_121 Depth=1
	incq	%rdx
	addq	$8192, %rcx                     ## imm = 0x2000
LBB0_121:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_124 Depth 2
	cmpq	$1023, %rdx                     ## imm = 0x3FF
	jg	LBB0_126
## %bb.122:                             ## %.preheader3
                                        ##   in Loop: Header=BB0_121 Depth=1
	leaq	(%rax,%rcx), %rsi
	xorl	%edi, %edi
	cmpq	$1023, %rdi                     ## imm = 0x3FF
	jg	LBB0_125
	.p2align	4
LBB0_124:                               ##   Parent Loop BB0_121 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	$0, (%rsi,%rdi,8)
	incq	%rdi
	cmpq	$1023, %rdi                     ## imm = 0x3FF
	jle	LBB0_124
	jmp	LBB0_125
LBB0_126:                               ## %.preheader2
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	LBB0_127
	.p2align	4
LBB0_134:                               ##   in Loop: Header=BB0_127 Depth=1
	incq	%rdx
	addq	$8192, %rcx                     ## imm = 0x2000
LBB0_127:                               ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_129 Depth 2
                                        ##       Child Loop BB0_132 Depth 3
	cmpq	$1023, %rdx                     ## imm = 0x3FF
	jg	LBB0_135
## %bb.128:                             ## %.preheader1
                                        ##   in Loop: Header=BB0_127 Depth=1
	movq	%rdx, %rsi
	shlq	$10, %rsi
	leaq	(%rbx,%rcx), %rdi
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	jmp	LBB0_129
	.p2align	4
LBB0_133:                               ##   in Loop: Header=BB0_129 Depth=2
	incq	%r9
	addq	$8, %r8
LBB0_129:                               ##   Parent Loop BB0_127 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_132 Depth 3
	cmpq	$1023, %r9                      ## imm = 0x3FF
	jg	LBB0_134
## %bb.130:                             ## %.preheader
                                        ##   in Loop: Header=BB0_129 Depth=2
	leaq	(%rsi,%r9), %r10
	movq	%r8, %r11
	xorl	%r15d, %r15d
	cmpq	$1023, %r15                     ## imm = 0x3FF
	jg	LBB0_133
	.p2align	4
LBB0_132:                               ##   Parent Loop BB0_127 Depth=1
                                        ##     Parent Loop BB0_129 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdi,%r15,8), %xmm0            ## xmm0 = mem[0],zero
	vmovsd	(%r14,%r11), %xmm1              ## xmm1 = mem[0],zero
	vfmadd213sd	(%rax,%r10,8), %xmm0, %xmm1 ## xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%rax,%r10,8)
	incq	%r15
	addq	$8192, %r11                     ## imm = 0x2000
	cmpq	$1023, %r15                     ## imm = 0x3FF
	jle	LBB0_132
	jmp	LBB0_133
LBB0_135:
	vmovsd	LCPI0_9(%rip), %xmm0            ## xmm0 = [1.0E+1,0.0E+0]
	callq	_printFloat
	xorl	%eax, %eax
	addq	$8, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
.subsections_via_symbols
